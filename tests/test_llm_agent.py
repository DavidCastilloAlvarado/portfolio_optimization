"""Tests for models/llm_agent.py (LangGraph agent engine; LLM + MCP mocked)."""

import numpy as np
import pandas as pd
import pytest
from config.defaults import Config
from config.llm import LLMSettings, WebSearchMCPSettings
import models.llm_agent as agent_mod
from models.llm_agent import ViewContract, build_user_prompt
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


class FakeAgent:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    async def ainvoke(self, messages, config=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"structured_response": outcome}


def _contract(value_pct, confidence=0.5, rationale="r", sources=None):
    return ViewContract(
        predicted_avg_daily_return_pct=value_pct,
        confidence=confidence,
        rationale=rationale,
        sources=sources or [],
    )


@pytest.fixture
def ret_frame():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2026-01-01", periods=61)
    a = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 61))
    prices = pd.DataFrame({"AAA": a}, index=dates)
    return prices.pct_change().dropna()


@pytest.fixture
def cov_frame(ret_frame):
    return np.array(ret_frame.cov())


class TestViewContract:
    def test_valid_contract(self):
        c = _contract(0.05, 0.4, "drivers", ["http://a.example"])
        assert c.predicted_avg_daily_return_pct == 0.05
        assert c.confidence == 0.4
        assert c.sources == ["http://a.example"]

    def test_negative_return_allowed(self):
        c = _contract(-0.12)
        assert c.predicted_avg_daily_return_pct == -0.12

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            _contract(0.05, confidence=1.5)
        with pytest.raises(Exception):
            _contract(0.05, confidence=-0.1)


class TestPrompts:
    def test_user_prompt_contents(self):
        cfg = Config(llm_view_engine="agent", blm_horizon_days=14)
        hist = np.array([0.1, -0.2, 0.3])
        prompt = build_user_prompt("GOOG", cfg, hist)
        assert "GOOG" in prompt
        assert "Past 3 daily returns" in prompt
        assert "14 trading days" in prompt
        assert "0.10, -0.20, 0.30" in prompt


class TestGenerateAgentViews:
    def test_samples_become_q_and_omega(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([
            FakeAgent([_contract(0.10), _contract(0.30), _contract(-0.10)]),
        ])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=3)

        views, from_cache = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        assert from_cache is False
        v = views["AAA"]
        assert v["q"] == pytest.approx(0.10 / 100)
        assert v["n_ok"] == 3
        assert v["omega"] == pytest.approx(4e-6, rel=1e-6)
        assert len(v["details"]) == 3
        assert v["details"][0]["rationale"] == "r"
        assert len(list(tmp_path.glob("llm_views_*.json"))) == 1

    def test_plausibility_clamp(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([
            FakeAgent([_contract(90.0), _contract(-120.0), _contract(0.20)]),
        ])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=3)

        views, _ = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        v = views["AAA"]
        assert v["n_ok"] == 1
        assert v["q"] == pytest.approx(0.002)
        assert v["omega"] == pytest.approx(agent_mod.OMEGA_FLOOR_FRACTION * cfg.llm_tau * cov_frame[0, 0])

    def test_agent_failure_falls_back_to_direct(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([
            FakeAgent([RuntimeError("agent exploded")]),
        ])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(
            agent_mod, "_sample_asset_direct",
            lambda name, hist, cfg, settings: [0.001, 0.0015],
        )
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=2)

        views, _ = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        v = views["AAA"]
        assert v["n_ok"] == 2
        assert v["q"] == pytest.approx(0.00125)
        assert "details" not in v

    def test_agent_retry_on_transient_failure(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([
            FakeAgent([RuntimeError("timeout"), _contract(0.20), _contract(0.40)]),
        ])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=2)

        views, _ = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        v = views["AAA"]
        assert v["n_ok"] == 2
        assert len(v["details"]) == 2

    def test_agent_double_failure_falls_back_to_direct(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([
            FakeAgent([RuntimeError("x"), RuntimeError("y")]),
        ])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(
            agent_mod, "_sample_asset_direct",
            lambda name, hist, cfg, settings: [0.001],
        )
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=1)

        views, _ = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        v = views["AAA"]
        assert v["n_ok"] == 1
        assert v["q"] == pytest.approx(0.001)
        assert "details" not in v

    def test_cache_reuse_skips_agent(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        agents = iter([FakeAgent([_contract(0.10)])])
        monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: next(agents))
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cfg = Config(llm_view_engine="agent", llm_samples=1)

        views1, from_cache1 = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        assert from_cache1 is False

        def explode(*a, **k):
            raise AssertionError("agent must not run again")

        monkeypatch.setattr(agent_mod, "create_agent", explode)
        views2, from_cache2 = agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        assert from_cache2 is True
        assert views2["AAA"]["q"] == pytest.approx(views1["AAA"]["q"])
        assert views2["AAA"]["details"] == views1["AAA"]["details"]


class TestResearchCacheTools:
    def test_load_missing_returns_no_cache_message(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        out = agent_mod.load_research.invoke({"ticker": "tsm"})
        assert "No cached research" in out
        assert not list(tmp_path.glob("*.md"))

    def test_save_then_load_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        agent_mod.save_research.invoke({"ticker": "tsm", "notes": "## News\n- revenue up\n"})
        files = list(tmp_path.glob("research_TSM_*.md"))
        assert len(files) == 1
        assert agent_mod.research_cache_path("TSM") == files[0]
        text = agent_mod.load_research.invoke({"ticker": "TSM"})
        assert text.startswith("# Research notes")
        assert "revenue up" in text

    def test_tools_registered_with_agent(self, monkeypatch, tmp_path, ret_frame, cov_frame):
        monkeypatch.setattr(agent_mod, "MCPAdapter", FakeAdapter)
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        agents = iter([FakeAgent([_contract(0.10)])])
        captured = {}

        def fake_create_agent(llm, tools, system_prompt=None, response_format=None):
            captured["tools"] = tools
            return next(agents)

        monkeypatch.setattr(agent_mod, "create_agent", fake_create_agent)
        cfg = Config(llm_view_engine="agent", llm_samples=1)
        agent_mod.generate_agent_views(
            ret_frame, ["AAA"], cov_frame, cfg,
            settings=SETTINGS, mcp_settings=MCP_SETTINGS,
        )
        tool_names = {t.name for t in captured["tools"]}
        assert {"load_research", "save_research"} <= tool_names

    def test_directive_covers_research_protocol(self):
        assert "load_research" in agent_mod.SYSTEM_DIRECTIVE
        assert "save_research" in agent_mod.SYSTEM_DIRECTIVE
        assert "do NOT repeat the web searches" in agent_mod.SYSTEM_DIRECTIVE

    def test_directive_demands_exhaustive_notes(self):
        assert "LONG, detailed, self-contained" in agent_mod.SYSTEM_DIRECTIVE
        assert "reconstructable" in agent_mod.SYSTEM_DIRECTIVE
        assert "ALL urls you actually used" in agent_mod.SYSTEM_DIRECTIVE
        assert "Bull case and" in agent_mod.SYSTEM_DIRECTIVE

    def test_user_prompt_mentions_protocol_and_date(self):
        cfg = Config(llm_view_engine="agent", blm_horizon_days=14)
        prompt = build_user_prompt("GOOG", cfg, np.array([0.1]))
        assert "load_research first" in prompt
        assert f"Today: {agent_mod._today_iso()}" in prompt
