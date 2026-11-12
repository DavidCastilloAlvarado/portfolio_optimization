"""Research agent for Black-Litterman views: LangGraph agent + MCP web search.

Each view is produced by a ReAct agent (LangChain `create_agent`) that must
research the ticker on the web — latest news, analyst consensus, earnings,
macro context — through a web-search MCP server before forecasting, and
returns the forecast as a strict Pydantic contract (`ViewContract`). The
agent model runs with reasoning enabled at `Config.llm_reasoning_effort`
(default "medium"). Research is cached per ticker per day as a long,
self-contained markdown note (`load_research` / `save_research` tools), so
only the first sample of a day pays for the web research. K agent samples
per asset give the view mean (q) and sample variance (Ω) exactly like the
direct engine; per-asset agent failures fall back to the direct
history-only sampler.
"""

import asyncio
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain.mcp import MCPAdapter

import core.data.cache as price_cache
from config.defaults import Config
from config.llm import LLMSettings, WebSearchMCPSettings
from core.progress import elapsed as _elapsed
from core.progress import progress_log
from models.blm import (
    MAX_PLAUSIBLE_DAILY_PCT,
    MIN_HISTORY_OBS,
    OMEGA_FLOOR_FRACTION,
    VIEW_TEMPERATURE,
    _read_cached_views,
    _sample_asset_direct,
    _sample_marker,
    _views_cache_path,
    _write_cached_views,
)

AGENT_SAMPLE_TIMEOUT_S = 1200.0
RESEARCH_CACHE_PREFIX = "research_"

SYSTEM_DIRECTIVE = (
    "You are a professional equity research analyst. Research protocol: FIRST call the "
    "load_research tool with the ticker. If it returns cached notes, treat them as the "
    "complete research record: use them as your research base and do NOT repeat the web "
    "searches. If it reports no cached research, research the asset exhaustively on the "
    "web: run MULTIPLE web searches with specific queries including the ticker symbol "
    "(news of the past two weeks, analyst ratings and individual price targets, the "
    "most recent earnings report and guidance, upcoming catalysts, macro and sector "
    "conditions), and use the page-fetch tool on several of the most authoritative "
    "results so you have full article content, not just snippets. Immediately after "
    "finishing, call the save_research tool with a LONG, detailed, self-contained "
    "markdown research note - NOT a summary. Any later forecast must be reconstructable "
    "from the note alone, without new research. Required structure: "
    "(1) Company snapshot; (2) Latest news: every relevant item from the past two weeks "
    "with dates; (3) Analyst consensus: each tracked firm with action, price target and "
    "date where available, plus consensus averages; (4) Latest earnings: all headline "
    "numbers verbatim (revenue, EPS, margins, segments) and the full guidance; "
    "(5) Upcoming catalysts with dates; (6) Macro/sector environment; (7) Bull case and "
    "bear case with concrete risks; (8) Sources: a numbered list of ALL urls you "
    "actually used, cited inline as [1], [2] throughout the note. Quote key figures "
    "verbatim rather than rounding or paraphrasing; prefer completeness over brevity. "
    "Then combine the research with the provided price history and forecast the AVERAGE "
    "daily return (percent per day) for the requested horizon. Be honest about "
    "uncertainty in 'confidence' (0-1). List ONLY urls you actually used in "
    "'sources'. Never invent sources or data."
)


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def research_cache_path(ticker: str, day: str | None = None) -> Path:
    if day is None:
        day = _today_iso()
    return Path(price_cache.CACHE_DIR) / f"{RESEARCH_CACHE_PREFIX}{ticker.upper()}_{day}.md"


@tool
def load_research(ticker: str) -> str:
    """Load today's cached web research notes for this ticker. Call this FIRST, before any web search. If markdown notes are returned, reuse them instead of searching again; if it reports no cached research, run the web research and save it with save_research."""
    path = research_cache_path(ticker)
    if not path.exists():
        return f"No cached research for {ticker.upper()} for today - run the web research and save it with the save_research tool."
    return path.read_text()


@tool
def save_research(ticker: str, notes: str) -> str:
    """Save the web research notes for this ticker for today as a markdown file, so later samples or runs on the same day can reuse them. Call this right after finishing the web research, before producing the forecast contract."""
    path = research_cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Research notes - {ticker.upper()} - {_today_iso()}\n\n"
    path.write_text(header + notes)
    return f"Saved {len(notes)} chars of research for {ticker.upper()} to {path.name}."


class _TurnLogger(BaseCallbackHandler):
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.t0 = time.monotonic()
        self.turns = 0
        self.tools = 0

    def on_llm_end(self, response, **kwargs) -> None:
        self.turns += 1
        progress_log(self.tag, f"llm turn {self.turns} done (+{_elapsed(self.t0)})")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.tools += 1
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        arg = (input_str or "").replace("\n", " ")[:220]
        progress_log(self.tag, f"tool #{self.tools} {name} :: {arg}")

    def on_tool_end(self, output, **kwargs) -> None:
        name = kwargs.get("name") or "tool"
        progress_log(self.tag, f"tool_result {name} ({len(str(output or ''))} chars)")

    def on_tool_error(self, error, **kwargs) -> None:
        name = kwargs.get("name") or "tool"
        progress_log(self.tag, f"tool_error {name} :: {str(error)[:160]}")


class ViewContract(BaseModel):
    predicted_avg_daily_return_pct: float = Field(
        description=(
            "Forecast AVERAGE daily return (percent per day) over the requested horizon; "
            "e.g. 0.10 means +0.10% per day"
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence in the forecast, 0-1")
    rationale: str = Field(description="1-3 sentences on the main drivers of the forecast")
    sources: list[str] = Field(default_factory=list, description="URLs actually used in the web research")


def build_llm(cfg: Config, settings: LLMSettings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key or "not-needed",
        model=settings.model,
        temperature=VIEW_TEMPERATURE,
        max_tokens=cfg.llm_agent_max_tokens,
        timeout=settings.timeout,
        extra_body={"reasoning_effort": cfg.llm_reasoning_effort},
    )


def build_user_prompt(
    name: str,
    cfg: Config,
    history_pct: np.ndarray,
    sample_index: int | None = None,
) -> str:
    hist = ", ".join(f"{v:.2f}" for v in history_pct)
    marker = "" if sample_index is None else _sample_marker(sample_index, cfg.llm_samples) + "\n"
    return (
        f"{marker}"
        f"Asset: {name}\n"
        f"Today: {_today_iso()}\n"
        f"Past {len(history_pct)} daily returns, percent per day, oldest first: [{hist}]\n"
        f"Follow the research protocol (load_research first, research the web only if "
        f"nothing is cached, save_research after new research), then forecast the "
        f"average daily return (percent per day) for the next {cfg.blm_horizon_days} "
        f"trading days as the contract."
    )


async def _run_agent_sample(agent, prompt: str, recursion_limit: int, tag: str) -> ViewContract:
    logger = _TurnLogger(tag)
    result = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": prompt},
            config={"recursion_limit": recursion_limit, "callbacks": [logger]},
        ),
        timeout=AGENT_SAMPLE_TIMEOUT_S,
    )
    progress_log(tag, f"agent finished ({logger.tools} tool calls, {logger.turns} turns, +{_elapsed(logger.t0)})")
    return result["structured_response"]


async def _generate_agent_views_async(
    returns: pd.DataFrame,
    names: list[str],
    cov_returns: np.ndarray,
    cfg: Config,
    settings: LLMSettings,
    mcp_settings: WebSearchMCPSettings,
) -> dict:
    cov_diag = np.diag(np.asarray(cov_returns, dtype=float))
    omega_floor = OMEGA_FLOOR_FRACTION * cfg.llm_tau * cov_diag
    returns_asc = returns.sort_index()
    llm = build_llm(cfg, settings)

    views: dict = {}
    async with MCPAdapter(mcp_settings.mcp_target()) as adapter:
        tools = [*await adapter.list_tools(), load_research, save_research]
        progress_log("views", f"MCP ready: {len(tools)} tools ([{', '.join(getattr(t, 'name', str(t)) for t in tools)}])")
        for j, name in enumerate(names):
            series = returns_asc[name].dropna()
            if len(series) < MIN_HISTORY_OBS:
                continue
            history_pct = series.tail(cfg.blm_history_days).to_numpy(dtype=float) * 100.0
            memo = research_cache_path(name)
            memo_note = f"memo {memo.stat().st_size / 1024:.1f}KB (reuse)" if memo.exists() else "no memo (fresh research)"
            progress_log("views", f"asset {j + 1}/{len(names)}: {name} | history {len(history_pct)}d | {memo_note} | {cfg.llm_samples} samples")
            agent = create_agent(llm, tools, system_prompt=SYSTEM_DIRECTIVE, response_format=ViewContract)

            samples: list[float] = []
            details: list[dict] = []
            for k in range(cfg.llm_samples):
                t0 = time.monotonic()
                tag = f"views {name}"
                prompt = build_user_prompt(name, cfg, history_pct, sample_index=k + 1)
                try:
                    contract = await _run_agent_sample(agent, prompt, cfg.llm_agent_recursion_limit, tag=tag)
                except Exception:
                    progress_log("views", f"{name} sample {k + 1}/{cfg.llm_samples} FAILED (+{_elapsed(t0)}) — retry 1")
                    try:
                        contract = await _run_agent_sample(agent, prompt, cfg.llm_agent_recursion_limit, tag=tag)
                    except Exception:
                        progress_log("views", f"{name} sample {k + 1}/{cfg.llm_samples} FAILED on retry — stopping sampling")
                        break
                value = float(contract.predicted_avg_daily_return_pct)
                if not np.isfinite(value) or abs(value) > MAX_PLAUSIBLE_DAILY_PCT:
                    progress_log("views", f"{name} sample {k + 1}/{cfg.llm_samples}: implausible value {value} — rejected")
                    continue
                progress_log("views", f"{name} sample {k + 1}/{cfg.llm_samples} ok: {value:+.3f}%/day conf={contract.confidence:.2f} (+{_elapsed(t0)})")
                samples.append(value / 100.0)
                details.append({
                    "rationale": contract.rationale,
                    "sources": list(contract.sources),
                    "confidence": float(contract.confidence),
                })
            if not samples:
                progress_log("views", f"{name}: no valid agent samples — falling back to direct engine")
                samples = _sample_asset_direct(name, history_pct, cfg, settings)
            if not samples:
                continue

            q = float(np.mean(samples))
            var = float(np.var(samples, ddof=1)) if len(samples) > 1 else float(omega_floor[j])
            views[name] = {
                "q": q,
                "omega": max(var, float(omega_floor[j])),
                "samples": [float(s) for s in samples],
                "n_ok": len(samples),
            }
            if details:
                views[name]["details"] = details

    return views


def generate_agent_views(
    returns: pd.DataFrame,
    names: list[str],
    cov_returns: np.ndarray,
    cfg: Config,
    settings: LLMSettings | None = None,
    mcp_settings: WebSearchMCPSettings | None = None,
) -> tuple[dict, bool]:
    """Run the research-agent view engine with the shared daily view cache.

    Args:
        returns:      Daily simple-returns DataFrame (index = dates, one column per asset).
        names:        Asset names (columns of `returns`).
        cov_returns:  Daily covariance matrix (for the confidence floor).
        cfg:          Pipeline config (samples, effort, horizon, history window).
        settings:     LLM endpoint settings; defaults to LLMSettings.from_env().
        mcp_settings: Web-search MCP settings; defaults to WebSearchMCPSettings.from_env().

    Returns:
        (views, from_cache) — same shape as models.blm.generate_views; agent-sourced
        views additionally carry per-sample `details` (rationale, sources, confidence).
    """
    if settings is None:
        settings = LLMSettings.from_env()
    if mcp_settings is None:
        mcp_settings = WebSearchMCPSettings.from_env()

    path = _views_cache_path(names, settings.model, cfg)
    cached = _read_cached_views(path)
    if cached is not None:
        progress_log("views", f"view cache hit ({Path(path).name}) — {len(cached)} assets, no agent run")
        return cached, True

    t0 = time.monotonic()
    views = asyncio.run(
        _generate_agent_views_async(returns, names, cov_returns, cfg, settings, mcp_settings)
    )
    _write_cached_views(path, settings.model, views)
    progress_log("views", f"view generation done: {len(views)}/{len(names)} assets (+{_elapsed(t0)})")
    return views, False
