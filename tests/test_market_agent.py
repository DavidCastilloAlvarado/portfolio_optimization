"""Tests for models/selection_agents.py (briefing + verdict agents; LLM + MCP mocked)."""

import asyncio

import pytest

from config.defaults import Config
from config.llm import LLMSettings, WebSearchMCPSettings
import models.selection_agents as sa
import models.llm_agent as bl_agent
import core.data.cache as cache_mod

SETTINGS = LLMSettings(base_url="http://test/v1", api_key="", model="test-model", timeout=5)
MCP_SETTINGS = WebSearchMCPSettings(command="true", args=(), chrome_path="x", profile_dir="y")


class FakeAdapter:
    def __init__(self, target):
        self.target = target

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return []


class FakeMessage:
    def __init__(self, type, content, tool_calls=None, name=""):
        self.type = type
        self.content = content
        self.tool_calls = tool_calls or []
        self.name = name


def _fake_messages():
    return [
        FakeMessage("system", "SYSTEM DIRECTIVE TEXT"),
        FakeMessage("human", "Today: 2026-01-01"),
        FakeMessage("ai", "", tool_calls=[{"name": "search", "args": {"query": "fed rate decision"}}]),
        FakeMessage("tool", "SEARCH RESULT BODY with url http://news.example/1", name="search"),
        FakeMessage("ai", [{"type": "text", "text": "final reasoning text"}]),
    ]


class FakeAgent:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def ainvoke(self, messages, config=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"structured_response": outcome, "messages": _fake_messages()}


def _briefing():
    return sa.MarketBriefing(
        regime_summary="Risk-on",
        macro_risks=["Fed holds"],
        trending_themes=["AI"],
        overall_market_risk=0.3,
        sectors=[sa.SectorView(sector="Energy", trend_score=0.7, regime="sustained",
                               driver="Oil up", fragility=0.4, conviction=0.6)],
    )


def _verdict(conviction=0.8, include=True):
    return sa.SelectionVerdict(include=include, conviction=conviction, thesis="t",
                               catalysts=[], overhangs=[], earnings_in_days=None)


class TestContracts:
    def test_briefing_valid(self):
        b = _briefing()
        assert b.overall_market_risk == 0.3
        assert b.sectors[0].regime == "sustained"

    def test_briefing_bounds(self):
        with pytest.raises(Exception):
            sa.SectorView(sector="X", trend_score=1.5, regime="sustained",
                          driver="d", fragility=0.0, conviction=0.5)
        with pytest.raises(Exception):
            sa.MarketBriefing(regime_summary="s", overall_market_risk=-0.1, sectors=[])

    def test_verdict_bounds(self):
        v = _verdict()
        assert v.earnings_in_days is None
        with pytest.raises(Exception):
            _verdict(conviction=1.4)
        with pytest.raises(Exception):
            sa.SelectionVerdict(include=True, conviction=0.5, thesis="t",
                                catalysts=[], overhangs=[], earnings_in_days=-1)


class TestTranscriptDump:
    def test_full_passthrough_written(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        path = sa._dump_transcript("briefing", {"messages": _fake_messages()})
        assert path.name.startswith("selection_trace_briefing_")
        text = path.read_text()
        assert "SYSTEM DIRECTIVE TEXT" in text
        assert "[tool_call] search" in text
        assert '"query": "fed rate decision"' in text
        assert "[tool_result] search" in text
        assert "SEARCH RESULT BODY with url http://news.example/1" in text
        assert "final reasoning text" in text

    def test_verdict_trace_names_the_ticker(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        path = sa._dump_transcript("verdict", {"messages": _fake_messages()}, "vlo")
        assert path.name.startswith("selection_trace_verdict_VLO_")

    def test_empty_messages_no_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        path = sa._dump_transcript("briefing", {})
        assert path.read_text().startswith("# Agent trace")


class TestBriefingCacheKeying:
    def test_paths_differ_by_candidate_set(self):
        full = sa._briefing_cache_path(["Energy", "Health Care", "Financials"])
        part = sa._briefing_cache_path(["Energy", "Health Care"])
        assert full != part

    def test_key_is_order_insensitive(self):
        assert sa._briefing_key(["A", "B"]) == sa._briefing_key(["B", "A"])

    def test_partial_run_cannot_consume_full_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        with open(sa._briefing_cache_path(["A", "B"]), "w") as f:
            f.write("{}")
        assert sa._read_cached_briefing(["A", "B"]) is not None
        assert sa._read_cached_briefing(["A"]) is None
        assert sa._read_cached_briefing(["A", "B", "C"]) is None


class TestSelectionLLMSettings:
    def test_timeout_widened(self):
        widened = sa._selection_llm_settings(SETTINGS)
        assert widened.timeout == sa.SELECTION_LLM_TIMEOUT_S
        assert widened.model == SETTINGS.model

    def test_existing_larger_timeout_kept(self):
        big = LLMSettings(base_url="u", api_key="", model="m", timeout=900)
        assert sa._selection_llm_settings(big).timeout == 900


class TestDirectives:
    def test_briefing_directive_mission(self):
        d = sa.BRIEFING_DIRECTIVE
        assert "10-18 web searches" in d
        assert "MACRO" in d and "GEOPOLITICS" in d
        assert "inflation" in d.lower() and "war" in d.lower()
        assert "save_briefing" in d and "load_briefing" in d
        assert "ALL urls you actually used" in d
        assert "fragility" in d

    def test_briefing_prompt_has_sectors_and_date(self):
        p = sa.briefing_prompt(["Energy", "Utilities"])
        assert "Energy" in p and "Utilities" in p
        assert f"Today: {sa._today_iso()}" in p

    def test_verdict_directive_reuses_research_protocol(self):
        d = sa.VERDICT_DIRECTIVE
        assert "load_research" in d and "save_research" in d
        assert "include=true only if" in d
        assert "ONE sentence" in d

    def test_quant_digest_contents(self):
        entry = {"momentum": 0.9, "quality": 0.7, "finance_score": 0.8, "trailing_pe": 31.4,
                 "revenue_growth": 0.12, "margins": 0.24,
                 "metrics": {"ret_1m": 0.05, "ret_3m": 0.1, "ret_6m": 0.2,
                             "off_high_52w": -0.03, "max_dd_3m": -0.1, "above_ma50": True}}
        d = sa._quant_digest(entry)
        assert "momentum_rank=0.90" in d
        assert "trailing_pe=31.4" in d
        assert "above_ma50=True" in d


class TestSharedResearchMemo:
    def test_selection_and_black_litterman_reuse_same_ticker_memo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        sa.save_research.invoke({"ticker": "AAA", "notes": "Selection research memo"})
        assert "Selection research memo" in bl_agent.load_research.invoke({"ticker": "AAA"})

        bl_agent.save_research.invoke({"ticker": "BBB", "notes": "Black-Litterman research memo"})
        assert "Black-Litterman research memo" in sa.load_research.invoke({"ticker": "BBB"})
        assert sa.load_research is bl_agent.load_research
        assert sa.save_research is bl_agent.save_research


class TestBriefingMemoTools:
    def test_load_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        out = sa.load_briefing.invoke({})
        assert "No cached market briefing" in out

    def test_save_then_load(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        sa.save_briefing.invoke({"notes": "## Macro\n- rates steady\n"})
        files = list(tmp_path.glob("selection_briefing_*.md"))
        assert len(files) == 1
        text = sa.load_briefing.invoke({})
        assert text.startswith("# Market briefing memo")
        assert "rates steady" in text


class TestRunMarketBriefing:
    def _patch(self, monkeypatch, tmp_path, agents):
        monkeypatch.setattr(sa, "MCPAdapter", FakeAdapter)
        monkeypatch.setattr(sa, "build_llm", lambda cfg, s: "fake-llm")
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        it = iter(agents)
        captured = {}

        def fake_create_agent(llm, tools, system_prompt=None, response_format=None):
            captured["tools"] = tools
            captured["prompt"] = llm
            return next(it)

        monkeypatch.setattr(sa, "create_agent", fake_create_agent)
        return captured

    def test_unconfigured_mcp_raises(self):
        unconfigured = WebSearchMCPSettings(command="", args=(), chrome_path="x", profile_dir="y")
        with pytest.raises(RuntimeError, match="web-search MCP not configured"):
            sa.run_market_briefing(["Energy"], Config(), settings=SETTINGS, mcp_settings=unconfigured)

    def test_happy_path_and_cache(self, monkeypatch, tmp_path):
        captured = self._patch(monkeypatch, tmp_path, [FakeAgent([_briefing()])])
        out = sa.run_market_briefing(["Energy", "Utilities"], Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert out == _briefing()
        assert list(tmp_path.glob("selection_briefing_*.json"))
        trace = list(tmp_path.glob("selection_trace_briefing_*.md"))
        assert len(trace) == 1
        assert "fed rate decision" in trace[0].read_text()
        tool_names = {t.name for t in captured["tools"]}
        assert {"load_briefing", "save_briefing"} <= tool_names

        it2 = iter([FakeAgent([_briefing()])])

        def fake_create_agent2(llm, tools, system_prompt=None, response_format=None):
            raise AssertionError("agent should not run again from cache")

        monkeypatch.setattr(sa, "create_agent", fake_create_agent2)
        again = sa.run_market_briefing(["Energy", "Utilities"], Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert again == _briefing()

    def test_retry_then_succeed(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path, [FakeAgent([TimeoutError("slow"), _briefing()])])
        out = sa.run_market_briefing(["Energy"], Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert out == _briefing()

    def test_double_failure_raises(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path, [FakeAgent([TimeoutError("slow"), TimeoutError("slow")])])
        with pytest.raises(TimeoutError):
            sa.run_market_briefing(["Energy"], Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)


class TestRunNameVerdicts:
    def _patch(self, monkeypatch, tmp_path, agents_by_ticker):
        monkeypatch.setattr(sa, "MCPAdapter", FakeAdapter)
        monkeypatch.setattr(sa, "build_llm", lambda cfg, s: "fake-llm")
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        captured = {}

        def fake_create_agent(llm, tools, system_prompt=None, response_format=None):
            captured["tools"] = tools
            agent = FakeAgent(agents_by_ticker.pop(0))
            return agent

        monkeypatch.setattr(sa, "create_agent", fake_create_agent)
        return captured

    def test_per_name_verdicts_and_cache(self, monkeypatch, tmp_path):
        candidates = [{"ticker": "AAA", "name": "A", "sector": "Energy", "finance_score": 0.9,
                       "trailing_pe": 20.0, "revenue_growth": 0.1, "margins": 0.2, "metrics": {}},
                      {"ticker": "BBB", "name": "B", "sector": "Energy", "finance_score": 0.8,
                       "trailing_pe": 15.0, "revenue_growth": 0.05, "margins": 0.15, "metrics": {}}]
        self._patch(monkeypatch, tmp_path, [[_verdict(0.9)], [_verdict(0.7)]])
        out = sa.run_name_verdicts(candidates, Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert set(out) == {"AAA", "BBB"}
        assert out["AAA"].conviction == 0.9
        assert list(tmp_path.glob("selection_verdict_AAA_*.json"))
        assert list(tmp_path.glob("selection_verdict_BBB_*.json"))
        assert list(tmp_path.glob("selection_trace_verdict_AAA_*.md"))
        assert list(tmp_path.glob("selection_trace_verdict_BBB_*.md"))

        def fake_create_agent2(*a, **k):
            raise AssertionError("cached verdicts must not run the agent")

        monkeypatch.setattr(sa, "create_agent", fake_create_agent2)
        again = sa.run_name_verdicts(candidates, Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert set(again) == {"AAA", "BBB"}

    def test_per_name_failure_skipped_others_continue(self, monkeypatch, tmp_path):
        candidates = [{"ticker": "AAA", "name": "A", "sector": "Energy", "finance_score": 0.9,
                       "trailing_pe": 20.0, "revenue_growth": 0.1, "margins": 0.2, "metrics": {}},
                      {"ticker": "BBB", "name": "B", "sector": "Energy", "finance_score": 0.8,
                       "trailing_pe": 15.0, "revenue_growth": 0.05, "margins": 0.15, "metrics": {}}]
        self._patch(monkeypatch, tmp_path,
                    [[RuntimeError("boom"), RuntimeError("boom")], [_verdict(0.7)]])
        out = sa.run_name_verdicts(candidates, Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        assert set(out) == {"BBB"}

    def test_tools_include_shared_research_protocol(self, monkeypatch, tmp_path):
        candidates = [{"ticker": "AAA", "name": "A", "sector": "Energy", "finance_score": 0.9,
                       "trailing_pe": 20.0, "revenue_growth": 0.1, "margins": 0.2, "metrics": {}}]
        captured = self._patch(monkeypatch, tmp_path, [[_verdict(0.9)]])
        sa.run_name_verdicts(candidates, Config(), settings=SETTINGS, mcp_settings=MCP_SETTINGS)
        tool_names = {t.name for t in captured["tools"]}
        assert {"load_research", "save_research"} <= tool_names

    def test_verdict_agents_respect_configured_concurrency(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sa, "MCPAdapter", FakeAdapter)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        active = 0
        peak_active = 0

        class ConcurrentAgent:
            async def ainvoke(self, messages, config=None):
                nonlocal active, peak_active
                active += 1
                peak_active = max(peak_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                return {"structured_response": _verdict(), "messages": _fake_messages()}

        monkeypatch.setattr(sa, "_build_verdict_agent", lambda *args: ConcurrentAgent())
        candidates = [
            {"ticker": ticker, "name": ticker, "sector": "Energy", "finance_score": 0.8, "metrics": {}}
            for ticker in ("AAA", "BBB", "CCC", "DDD")
        ]
        out = sa.run_name_verdicts(
            candidates,
            Config(selection_verdict_concurrency=2),
            settings=SETTINGS,
            mcp_settings=MCP_SETTINGS,
        )
        assert set(out) == {"AAA", "BBB", "CCC", "DDD"}
        assert peak_active == 2

    def test_unconfigured_mcp_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        unconfigured = WebSearchMCPSettings(command="", args=(), chrome_path="x", profile_dir="y")
        with pytest.raises(RuntimeError, match="web-search MCP not configured"):
            sa.run_name_verdicts([], Config(), settings=SETTINGS, mcp_settings=unconfigured)
