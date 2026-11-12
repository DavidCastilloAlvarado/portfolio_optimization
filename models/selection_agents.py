"""LLM research agents for the S&P 100 selection funnel.

Two agents, both on the LangChain `create_agent` + web-search MCP protocol of
models/llm_agent.py (one shared MCP session, per-call timeout, one retry,
strict Pydantic contracts, daily caches):

- Market briefing: a chief-strategist agent that runs MANY web searches across
  macro (rates, inflation, labor), geopolitics (war, sanctions, energy), the
  technology/AI industry, each candidate sector, and market sentiment, writes
  a long self-contained memo (load_briefing / save_briefing tools, one per day),
  and returns a MarketBriefing with a per-sector regime call.
- Name verdict: one pass per candidate name that reuses the SHARED per-ticker
  research memo protocol of the BL agent (load_research / save_research from
  models/llm_agent.py — a memo written by either feature is reused by the
  other), does a few targeted searches only when no memo exists, and returns a
  SelectionVerdict (include, conviction, thesis, catalysts, overhangs,
  earnings distance).
"""

import asyncio
import dataclasses
import hashlib
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain.mcp import MCPAdapter

import core.data.cache as price_cache
from config.defaults import Config
from config.llm import LLMSettings, WebSearchMCPSettings
from core.progress import elapsed, progress_log
from models.llm_agent import (
    AGENT_SAMPLE_TIMEOUT_S,
    _TurnLogger,
    _today_iso,
    build_llm,
    load_research,
    research_cache_path,
    save_research,
)

BRIEFING_CACHE_PREFIX = "selection_briefing_"
VERDICT_CACHE_PREFIX = "selection_verdict_"
SELECTION_LLM_TIMEOUT_S = 600.0


def _briefing_key(candidates: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(candidates)).encode()).hexdigest()[:8]


class SectorView(BaseModel):
    sector: str = Field(description="Exact GICS sector name from the request")
    trend_score: float = Field(ge=0.0, le=1.0, description="How strongly this sector is trending right now, 0-1")
    regime: str = Field(description="One of: accelerating, sustained, fading, choppy")
    driver: str = Field(description="1-2 sentences: what is driving the sector move (real events, dates)")
    fragility: float = Field(ge=0.0, le=1.0, description="1 = narrow/fragile move (few names, hype-driven), 0 = broad and solid")
    conviction: float = Field(ge=0.0, le=1.0, description="Honest confidence in the regime call, 0-1")


class MarketBriefing(BaseModel):
    regime_summary: str = Field(description="3-5 sentences: the overall market regime right now")
    macro_risks: list[str] = Field(default_factory=list, description="Concrete macro risks with dates/figures")
    trending_themes: list[str] = Field(default_factory=list, description="Themes with real momentum right now")
    overall_market_risk: float = Field(ge=0.0, le=1.0, description="1 = strongly risk-off environment, 0 = calm risk-on")
    sectors: list[SectorView] = Field(default_factory=list)


class SelectionVerdict(BaseModel):
    include: bool = Field(description="True only if fundamentals/news support holding this name in a 10-name trending portfolio")
    conviction: float = Field(ge=0.0, le=1.0, description="Honest conviction, 0-1; below 0.5 when evidence is thin")
    thesis: str = Field(description="ONE sentence a portfolio manager could quote")
    catalysts: list[str] = Field(default_factory=list, description="Upcoming positive catalysts with dates")
    overhangs: list[str] = Field(default_factory=list, description="Concrete risks (earnings date, litigation, regulation, valuation, dilution)")
    earnings_in_days: int | None = Field(default=None, ge=0, le=90, description="Trading days until next earnings, if established, else null")


def _briefing_memo_path(day: str | None = None) -> Path:
    if day is None:
        day = _today_iso()
    return Path(price_cache.CACHE_DIR) / f"{BRIEFING_CACHE_PREFIX}{day}.md"


def _briefing_cache_path(candidates: list[str], day: str | None = None) -> str:
    if day is None:
        day = _today_iso()
    return f"{price_cache.CACHE_DIR}/{BRIEFING_CACHE_PREFIX}{_briefing_key(candidates)}_{day}.json"


def _verdict_cache_path(ticker: str, day: str | None = None) -> str:
    if day is None:
        day = _today_iso()
    return f"{price_cache.CACHE_DIR}/{VERDICT_CACHE_PREFIX}{ticker.upper()}_{day}.json"


@tool
def load_briefing() -> str:
    """Load today's cached market briefing memo (the previous strategist run of the day). Call this FIRST: if a memo is returned, build on it and do NOT repeat all the web searches; if it reports no cached briefing, run the full research protocol."""
    path = _briefing_memo_path()
    if not path.exists():
        return "No cached market briefing for today - run the full research protocol and save it with the save_briefing tool."
    return path.read_text()


@tool
def save_briefing(notes: str) -> str:
    """Save the market briefing memo for today as a markdown file so later runs of the day can build on it. Call this right after finishing the web research, before producing the briefing contract."""
    path = _briefing_memo_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Market briefing memo - {_today_iso()}\n\n"
    path.write_text(header + notes)
    return f"Saved {len(notes)} chars of market briefing to {path.name}."


BRIEFING_DIRECTIVE = (
    "You are a chief market strategist. Mission: produce an evidence-backed market "
    "briefing that will be used to select the most trending sectors and names in the "
    "S&P 100. Step 0: FIRST call load_briefing. If today's memo is returned, build "
    "on it and run only the additional targeted searches needed to fill gaps or "
    "refresh stale facts; if it reports no cached briefing, run the full protocol "
    "below. Research protocol (MANDATORY, in this order): (1) MACRO - at least 2 "
    "web searches with different phrasings on interest rates / central bank policy, "
    "the latest inflation data, and the labor market; (2) GEOPOLITICS - at least 1-2 "
    "searches on active wars or conflicts that affect markets, sanctions, and energy "
    "supply; (3) TECHNOLOGY / AI INDUSTRY - at least 2 searches on AI capex, "
    "semiconductors, data centers and the biggest recent technology earnings; "
    "(4) EACH candidate sector named in the request - at least 2 searches per sector "
    "(recent news plus drivers and risks); (5) MARKET SENTIMENT - at least 1 search on "
    "VIX, market breadth and risk-on / risk-off flows. Run 10-18 web searches in "
    "total, varying the phrasing of your queries so you get a real cross-section of "
    "the news instead of one article; use the page-fetch tool on the most "
    "authoritative results when a snippet is not enough. Afterwards, FIRST call "
    "save_briefing with a LONG, detailed, self-contained markdown memo of everything "
    "you found - NOT a summary: dated headlines with figures, per-sector findings, "
    "and a numbered Sources list of ALL urls you actually used, cited inline as [1], "
    "[2] throughout. Prefer completeness over brevity. Quote real dates, numbers and "
    "named events; when evidence for a claim is thin, lower the conviction instead of "
    "guessing. Then answer with the structured briefing: one SectorView for EVERY "
    "candidate sector, each with an honest regime call (accelerating, sustained, "
    "fading or choppy), a fragility score (1 = narrow/hype-driven move carried by a "
    "few names, 0 = broad and solid) and a driver grounded in the searched news. "
    "Never invent sources, dates or data."
)


def briefing_prompt(candidate_sectors: list[str]) -> str:
    return (
        f"Today: {_today_iso()}\n"
        f"Candidate sectors from the quant screen, best momentum first: "
        f"{', '.join(candidate_sectors)}\n"
        f"Follow the research protocol (load_briefing first, full multi-search "
        f"protocol when nothing is cached, save_briefing after new research), then "
        f"produce the market briefing with one SectorView for every candidate sector."
    )


def _selection_llm_settings(settings: LLMSettings) -> LLMSettings:
    """Per-request LLM timeout widened for the selection agents.

    Their context accumulates fetched articles, so a single reasoning step on
    the local model can exceed the default 120 s request timeout; the overall
    per-agent-call cap (AGENT_SAMPLE_TIMEOUT_S) still bounds the run.
    """
    return dataclasses.replace(settings, timeout=max(settings.timeout, SELECTION_LLM_TIMEOUT_S))


def _message_text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def _dump_transcript(kind: str, result: dict, ticker: str | None = None) -> Path:
    """Write the agent's FULL pass-through to temp/ as markdown.

    The agent result carries the whole conversation in `result["messages"]`:
    the system directive, the prompt, every tool call with its exact query,
    every tool result (full fetched-article text), and the final answer.
    Files: temp/selection_trace_briefing_{date}.md and
    temp/selection_trace_verdict_{TICKER}_{date}.md
    """
    day = _today_iso()
    suffix = f"_{ticker.upper()}" if ticker else ""
    title = f"Agent trace - {kind}" + (f" {ticker.upper()}" if ticker else "") + f" - {day}"
    path = Path(price_cache.CACHE_DIR) / f"selection_trace_{kind}{suffix}_{day}.md"
    lines = [f"# {title}\n"]
    for msg in result.get("messages", []):
        mtype = msg.type
        if mtype == "ai":
            for tc in getattr(msg, "tool_calls", None) or []:
                args = json.dumps(tc.get("args", {}), ensure_ascii=False, default=str)
                lines.append(f"\n## [tool_call] {tc.get('name')}\n{args}\n")
            text = _message_text(msg.content)
            if text.strip():
                lines.append(f"\n## [assistant]\n{text}\n")
        elif mtype == "tool":
            lines.append(f"\n## [tool_result] {getattr(msg, 'name', '') or 'tool'}\n{_message_text(msg.content)}\n")
        elif mtype == "human":
            lines.append(f"\n## [user]\n{_message_text(msg.content)}\n")
        elif mtype == "system":
            lines.append(f"\n## [system]\n{_message_text(msg.content)}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return path


def _build_briefing_agent(cfg: Config, settings: LLMSettings, tools: list):
    return create_agent(
        build_llm(cfg, _selection_llm_settings(settings)), tools,
        system_prompt=BRIEFING_DIRECTIVE, response_format=MarketBriefing,
    )


async def _run_briefing_async(
    candidate_sectors: list[str],
    cfg: Config,
    settings: LLMSettings,
    mcp_settings: WebSearchMCPSettings,
) -> dict:
    async with MCPAdapter(mcp_settings.mcp_target()) as adapter:
        tools = [*await adapter.list_tools(), load_briefing, save_briefing]
        progress_log("briefing", f"MCP ready: {len(tools)} tools | {len(candidate_sectors)} candidate sectors")
        agent = _build_briefing_agent(cfg, settings, tools)
        prompt = briefing_prompt(candidate_sectors)
        memo = _briefing_memo_path()
        progress_log("briefing", f"briefing agent start | memo {'yes (build on it)' if memo.exists() else 'none (full protocol)'}")
        logger = _TurnLogger("briefing")
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": prompt},
                    config={"recursion_limit": cfg.selection_recursion_limit, "callbacks": [logger]},
                ),
                timeout=AGENT_SAMPLE_TIMEOUT_S,
            )
        except Exception:
            progress_log("briefing", f"briefing agent FAILED (+{elapsed(t0)}) — retry 1")
            result = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": prompt},
                    config={"recursion_limit": cfg.selection_recursion_limit, "callbacks": [logger]},
                ),
                timeout=AGENT_SAMPLE_TIMEOUT_S,
            )
        progress_log("briefing", f"briefing agent done ({logger.tools} tool calls, {logger.turns} turns, +{elapsed(t0)})")
        _dump_transcript("briefing", result)
        return result


def _read_cached_briefing(candidates: list[str]) -> dict | None:
    path = _briefing_cache_path(candidates)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def run_market_briefing(
    candidate_sectors: list[str],
    cfg: Config,
    settings: LLMSettings | None = None,
    mcp_settings: WebSearchMCPSettings | None = None,
    force_refresh: bool = False,
) -> MarketBriefing:
    """Run the market-briefing agent (sync facade; cached once per day).

    Raises RuntimeError when the agent fails twice — the caller degrades to the
    quant-only screen.
    """
    if settings is None:
        settings = LLMSettings.from_env()
    if mcp_settings is None:
        mcp_settings = WebSearchMCPSettings.from_env()
    if not mcp_settings.is_configured:
        raise RuntimeError("web-search MCP not configured - market briefing unavailable")

    if not force_refresh:
        cached = _read_cached_briefing(candidate_sectors)
        if cached is not None:
            progress_log("briefing", "briefing cache hit — reusing today's briefing, no agent run")
            return MarketBriefing.model_validate(cached)

    result = asyncio.run(_run_briefing_async(candidate_sectors, cfg, settings, mcp_settings))
    briefing = result["structured_response"]
    os.makedirs(price_cache.CACHE_DIR, exist_ok=True)
    with open(_briefing_cache_path(candidate_sectors), "w") as f:
        json.dump(briefing.model_dump(), f)
    return briefing


VERDICT_DIRECTIVE = (
    "You are a professional equity analyst making an inclusion decision for a "
    "10-name trending portfolio. Protocol: FIRST call the load_research tool with "
    "the ticker. If it returns cached notes, treat them as the complete research "
    "record and do NOT repeat the web searches. If it reports no cached research, "
    "run 2-5 targeted web searches (latest news of the past two weeks, most recent "
    "earnings and guidance, analyst actions, upcoming events and catalysts) and "
    "then call save_research with a LONG, detailed, self-contained markdown note: "
    "dated headlines with figures, the key numbers verbatim, upcoming catalysts "
    "with dates, bull and bear case, and a numbered Sources list of ALL urls you "
    "actually used, cited inline. Prefer completeness over brevity. Then decide: "
    "include=true only if the fundamentals and news support the price momentum "
    "(earnings trajectory, real catalysts, no major overhang); a trend driven "
    "purely by sentiment without fundamental support gets include=false or "
    "conviction below 0.4. List concrete overhangs (earnings date, litigation, "
    "regulation, product risk, dilution, valuation extremes) and the trading "
    "distance to the next earnings report when you can establish it. The thesis "
    "must be ONE sentence a portfolio manager could quote. Be honest: conviction "
    "below 0.5 when your evidence is thin. Never invent sources, dates or data."
)


def _build_verdict_agent(cfg: Config, settings: LLMSettings, tools: list):
    return create_agent(
        build_llm(cfg, _selection_llm_settings(settings)), tools,
        system_prompt=VERDICT_DIRECTIVE, response_format=SelectionVerdict,
    )


def _quant_digest(name_entry: dict) -> str:
    m = name_entry.get("metrics", {})
    pe = name_entry.get("trailing_pe")
    pe_txt = f"{pe:.1f}" if pe else "n/a"
    return (
        f"momentum_rank={name_entry.get('momentum', 0):.2f} "
        f"quality_rank={name_entry.get('quality', 0):.2f} "
        f"finance_score={name_entry.get('finance_score', 0):.2f} "
        f"ret_1m={m.get('ret_1m', 0) * 100:+.1f}% ret_3m={m.get('ret_3m', 0) * 100:+.1f}% "
        f"ret_6m={m.get('ret_6m', 0) * 100:+.1f}% off_52w_high={m.get('off_high_52w', 0) * 100:.1f}% "
        f"max_dd_3m={m.get('max_dd_3m', 0) * 100:.1f}% above_ma50={m.get('above_ma50')} "
        f"trailing_pe={pe_txt} revenue_growth={name_entry.get('revenue_growth', 0) * 100:+.1f}% "
        f"margins={name_entry.get('margins', 0) * 100:.1f}%"
    )


async def _run_verdicts_async(
    candidates: list[dict],
    cfg: Config,
    settings: LLMSettings,
    mcp_settings: WebSearchMCPSettings,
) -> dict[str, SelectionVerdict]:
    async with MCPAdapter(mcp_settings.mcp_target()) as adapter:
        tools = [*await adapter.list_tools(), load_research, save_research]
        progress_log(
            "verdicts",
            f"MCP ready: {len(tools)} tools | {len(candidates)} candidates | "
            f"concurrency={cfg.selection_verdict_concurrency}",
        )
        semaphore = asyncio.Semaphore(cfg.selection_verdict_concurrency)

        async def run_one(index: int, entry: dict) -> tuple[str, SelectionVerdict | None]:
            ticker = entry["ticker"]
            async with semaphore:
                memo = research_cache_path(ticker)
                memo_note = f"memo {memo.stat().st_size / 1024:.1f}KB (reuse)" if memo.exists() else "no memo (fresh research)"
                progress_log("verdicts", f"name {index + 1}/{len(candidates)}: {ticker} | {memo_note}")
                prompt = (
                    f"Today: {_today_iso()}\nTicker: {ticker}\nName: {entry.get('name', '')}\n"
                    f"Sector: {entry.get('sector', '')}\nQuant digest: {_quant_digest(entry)}\n"
                    f"Follow the protocol (load_research first, targeted searches only when "
                    f"nothing is cached, save_research after new research), then make the "
                    f"inclusion decision for {ticker} as the contract."
                )
                logger = _TurnLogger(f"verdict {ticker}")
                t0 = time.monotonic()
                try:
                    agent = _build_verdict_agent(cfg, settings, tools)
                except Exception as exc:
                    progress_log("verdicts", f"{ticker} FAILED — name stays quant-only ({exc})")
                    return ticker, None
                for attempt in range(2):
                    try:
                        result = await asyncio.wait_for(
                            agent.ainvoke(
                                {"messages": prompt},
                                config={"recursion_limit": cfg.llm_agent_recursion_limit, "callbacks": [logger]},
                            ),
                            timeout=AGENT_SAMPLE_TIMEOUT_S,
                        )
                        break
                    except Exception:
                        if attempt == 0:
                            progress_log("verdicts", f"{ticker} FAILED (+{elapsed(t0)}) — retry 1")
                else:
                    progress_log("verdicts", f"{ticker} FAILED on retry — name stays quant-only")
                    return ticker, None
                try:
                    verdict = result["structured_response"]
                    _dump_transcript("verdict", result, ticker)
                except Exception as exc:
                    progress_log("verdicts", f"{ticker} result failed — name stays quant-only ({exc})")
                    return ticker, None
                progress_log(
                    "verdicts",
                    f"{ticker} done: include={verdict.include} conviction={verdict.conviction:.2f} "
                    f"({logger.tools} tools, +{elapsed(t0)})",
                )
                return ticker, verdict

        results = await asyncio.gather(*(run_one(i, entry) for i, entry in enumerate(candidates)))
    return {ticker: verdict for ticker, verdict in results if verdict is not None}


def _read_cached_verdict(ticker: str) -> dict | None:
    path = _verdict_cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def run_name_verdicts(
    candidates: list[dict],
    cfg: Config,
    settings: LLMSettings | None = None,
    mcp_settings: WebSearchMCPSettings | None = None,
    force_refresh: bool = False,
) -> dict[str, SelectionVerdict]:
    """Run a verdict pass per candidate name (sync facade; cached per name per day).

    Per-name failures are skipped (missing from the result) so the quant score
    still stands; a fully broken MCP session raises for the caller to degrade on.
    """
    if settings is None:
        settings = LLMSettings.from_env()
    if mcp_settings is None:
        mcp_settings = WebSearchMCPSettings.from_env()
    if not mcp_settings.is_configured:
        raise RuntimeError("web-search MCP not configured - name verdicts unavailable")

    pending = []
    for entry in candidates:
        ticker = entry["ticker"]
        if not force_refresh:
            cached = _read_cached_verdict(ticker)
            if cached is not None:
                continue
        pending.append(entry)
    progress_log(
        "verdicts",
        f"verdict pass: {len(candidates)} candidates | {len(pending)} fresh | {len(candidates) - len(pending)} cached",
    )

    fresh = asyncio.run(_run_verdicts_async(pending, cfg, settings, mcp_settings)) if pending else {}

    out: dict[str, SelectionVerdict] = {}
    for entry in candidates:
        ticker = entry["ticker"]
        if ticker in fresh:
            out[ticker] = fresh[ticker]
            os.makedirs(price_cache.CACHE_DIR, exist_ok=True)
            with open(_verdict_cache_path(ticker), "w") as f:
                json.dump(fresh[ticker].model_dump(), f)
        else:
            cached = _read_cached_verdict(ticker)
            if cached is not None:
                out[ticker] = SelectionVerdict.model_validate(cached)
    return out
