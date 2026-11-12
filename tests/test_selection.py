"""Tests for core/selection.py (quant funnel) and models/selection.py (orchestrator)."""

import json

import numpy as np
import pandas as pd
import pytest

from config.defaults import Config
import core.data.cache as cache_mod
import core.selection as core_sel
import models.selection as sel_mod
from models.selection_agents import MarketBriefing, SelectionVerdict, SectorView


def _prices_series(n=300, drift=0.001, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n, tz="UTC")
    vals = 100 * np.cumprod(1 + drift + rng.normal(0, 0.005, n))
    return pd.Series(vals, index=idx)


def _up_ticker():
    close = _prices_series(drift=0.002)
    vol = pd.Series(1e6, index=close.index)
    return close, vol


def _screen():
    def name(t, sector, fin, deduped=False, **m):
        metrics = {
            "ret_1m": 0.05, "ret_3m": 0.10, "ret_6m": 0.15,
            "off_high_52w": -0.02, "max_dd_3m": -0.08, "vol_3m": 0.01,
            "above_ma50": True, "rel_volume": 1.2,
        }
        metrics.update(m)
        return {
            "ticker": t, "name": f"{t} Corp", "sector": sector, "sub_industry": "Sub",
            "market_cap": 1e12, "trailing_pe": 30.0, "revenue_growth": 0.15,
            "margins": 0.25, "momentum": round(fin * 0.8, 4), "quality": round(fin * 0.5, 4),
            "finance_score": fin, "metrics": metrics, "deduped": deduped,
        }

    names = [
        name("A1", "Information Technology", 0.95),
        name("A2", "Information Technology", 0.90),
        name("A3", "Information Technology", 0.85),
        name("B1", "Financials", 0.80),
        name("C1", "Energy", 0.70),
        name("C2", "Energy", 0.65),
        name("D1", "Utilities", 0.60),
        name("X9", "Utilities", 0.99, deduped=True),
    ]
    sectors = [
        {"sector": "Information Technology", "n_members": 3, "momentum": 0.9, "breadth": 0.8,
         "dispersion": 0.1, "quant_score": 0.9},
        {"sector": "Financials", "n_members": 2, "momentum": 0.7, "breadth": 0.7,
         "dispersion": 0.1, "quant_score": 0.7},
        {"sector": "Energy", "n_members": 2, "momentum": 0.5, "breadth": 0.5,
         "dispersion": 0.1, "quant_score": 0.5},
        {"sector": "Utilities", "n_members": 2, "momentum": 0.2, "breadth": 0.2,
         "dispersion": 0.1, "quant_score": 0.2},
    ]
    return {"as_of": "2026-01-01", "universe_size": 100, "screened": 8, "skipped": [],
            "sectors": sectors, "names": names}


class TestLoadUniverse:
    def test_real_file(self):
        u = core_sel.load_universe()
        assert len(u) == 100
        assert u["NVDA"]["sector"] == "Information Technology"
        sectors = {e["sector"] for e in u.values()}
        assert sectors <= set(core_sel.GICS_SECTORS)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            core_sel.load_universe(str(tmp_path / "nope.json"))

    def test_bad_sector_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"AAA": {"name": "A", "sector": "Quantum", "sub_industry": "X"}}))
        with pytest.raises(ValueError, match="unknown GICS sector"):
            core_sel.load_universe(str(p))


class TestComputeNameMetrics:
    def test_uptrend_metrics(self):
        idx = pd.bdate_range("2025-01-01", periods=300, tz="UTC")
        close = pd.Series(100 * (1.003 ** np.arange(300)), index=idx)
        vol = pd.Series(1e6, index=idx)
        m = core_sel.compute_name_metrics(close, vol)
        assert m is not None
        assert m["ret_3m"] > 0.1
        assert m["off_high_52w"] == pytest.approx(0.0)
        assert m["max_dd_3m"] == pytest.approx(0.0)
        assert m["rel_volume"] == pytest.approx(1.0)
        assert m["above_ma50"] is True

    def test_short_history_returns_none(self):
        close, _ = _up_ticker()
        assert core_sel.compute_name_metrics(close.iloc[:30]) is None


class TestScoring:
    def test_momentum_ordering(self):
        metrics = {"HOT": {"ret_1m": 0.1, "ret_3m": 0.2, "ret_6m": 0.3, "off_high_52w": 0.0, "rel_volume": 2.0},
                   "COLD": {"ret_1m": -0.05, "ret_3m": -0.1, "ret_6m": -0.2, "off_high_52w": -0.3, "rel_volume": 0.5}}
        quotes = {"HOT": {"epsTrailingTwelveMonths": 5.0, "trailingPE": 30.0},
                  "COLD": {"epsTrailingTwelveMonths": 1.0, "trailingPE": 10.0}}
        scored = core_sel.score_universe(metrics, quotes)
        assert scored["HOT"]["momentum"] > scored["COLD"]["momentum"]
        assert scored["HOT"]["finance"] > scored["COLD"]["finance"]

    def test_quality_requires_earnings(self):
        metrics = {"X": {"ret_1m": 0.01, "ret_3m": 0.02, "ret_6m": 0.03, "off_high_52w": -0.01, "rel_volume": 1.0},
                   "Y": {"ret_1m": 0.01, "ret_3m": 0.02, "ret_6m": 0.03, "off_high_52w": -0.01, "rel_volume": 1.0}}
        quotes = {"X": {"trailingAnnualRevenueGrowth": 0.2, "profitMargins": None,
                        "epsTrailingTwelveMonths": None, "trailingPE": None},
                  "Y": {"trailingAnnualRevenueGrowth": 0.2, "profitMargins": 0.3,
                        "epsTrailingTwelveMonths": None, "trailingPE": None}}
        scored = core_sel.score_universe(metrics, quotes)
        assert scored["X"]["quality"] == 0.0
        assert scored["Y"]["quality"] > 0.0


class TestSectorScores:
    def test_aggregation_and_ordering(self):
        universe = {"A": {"name": "A", "sector": "Energy", "sub_industry": "s"},
                    "B": {"name": "B", "sector": "Energy", "sub_industry": "s"},
                    "C": {"name": "C", "sector": "Utilities", "sub_industry": "s"}}
        metrics = {"A": {"ret_1m": 0.05, "ret_3m": 0.2, "off_high_52w": 0.0},
                   "B": {"ret_1m": 0.04, "ret_3m": 0.15, "off_high_52w": -0.01},
                   "C": {"ret_1m": -0.01, "ret_3m": -0.05, "off_high_52w": -0.2}}
        scored = {s: {"momentum": 0.5, "quality": 0.5, "finance": 0.5} for s in metrics}
        sectors = core_sel.sector_scores(scored, universe, metrics)
        assert sectors[0]["sector"] == "Energy"
        assert sectors[0]["breadth"] == pytest.approx(1.0)
        assert sectors[1]["sector"] == "Utilities"
        assert sectors[1]["breadth"] == pytest.approx(0.0)


class TestDedup:
    def test_twin_removed_smaller_cap(self):
        idx = pd.bdate_range("2025-01-01", periods=200, tz="UTC")
        vals = 100 * np.cumprod(1 + 0.001 + np.random.default_rng(3).normal(0, 0.005, 200))
        prices = pd.DataFrame({"GOOG": vals, "GOOGL": vals * 0.99, "MSFT": vals[::-1]}, index=idx)
        universe = {"GOOG": {"name": "G", "sector": "Communication Services", "sub_industry": "Internet"},
                    "GOOGL": {"name": "GL", "sector": "Communication Services", "sub_industry": "Internet"},
                    "MSFT": {"name": "M", "sector": "Information Technology", "sub_industry": "Software"}}
        quotes = {"GOOG": {"marketCap": 4e12}, "GOOGL": {"marketCap": 3e12}, "MSFT": {"marketCap": 3.5e12}}
        removed = core_sel._dedup_near_duplicates(prices, universe, quotes)
        assert removed == {"GOOGL"}

    def test_different_subindustry_not_removed(self):
        idx = pd.bdate_range("2025-01-01", periods=200, tz="UTC")
        vals = 100 * np.cumprod(1 + 0.001)
        prices = pd.DataFrame({"A": vals, "B": vals}, index=idx)
        universe = {"A": {"name": "A", "sector": "Energy", "sub_industry": "Oil"},
                    "B": {"name": "B", "sector": "Energy", "sub_industry": "Gas"}}
        removed = core_sel._dedup_near_duplicates(prices, universe, {})
        assert removed == set()


def _synth_universe_prices():
    idx = pd.bdate_range("2025-01-01", periods=300, tz="UTC")
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"Date": idx})
    drifts = {"T1": 0.003, "T2": 0.0025, "T3": 0.002, "F1": 0.001, "E1": 0.0, "U1": -0.001}
    for t, d in drifts.items():
        df[t] = 100 * np.cumprod(1 + d + rng.normal(0, 0.005, 300))
        df[f"{t}_volume"] = 1e6
    return df.set_index("Date")


class TestRunScreen:
    def test_structure_ranking_and_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        prices = _synth_universe_prices()
        calls = {"n": 0}

        def fake_prices(universe):
            calls["n"] += 1
            return prices, []

        quotes = {t: {"marketCap": 1e12, "trailingPE": 25.0, "epsTrailingTwelveMonths": 4.0,
                      "trailingAnnualRevenueGrowth": 0.1, "profitMargins": 0.2} for t in prices.columns
                  if not t.endswith("_volume")}
        monkeypatch.setattr(core_sel, "load_universe_prices", fake_prices)
        monkeypatch.setattr("core.data.quotes.fetch_batch_quotes", lambda syms: quotes)
        monkeypatch.setattr(core_sel, "load_universe", lambda path=None: {
            t: {"name": t, "sector": {"T1": "Information Technology", "T2": "Information Technology",
                                      "T3": "Information Technology", "F1": "Financials",
                                      "E1": "Energy", "U1": "Utilities"}[t], "sub_industry": "s"}
            for t in ("T1", "T2", "T3", "F1", "E1", "U1")
        })

        screen = core_sel.run_screen(Config())
        assert screen["from_cache"] is False
        assert screen["screened"] == 6
        assert screen["names"][0]["ticker"] in ("T1", "T2", "T3")
        assert screen["names"][-1]["ticker"] in ("U1", "E1")
        assert screen["sectors"][0]["sector"] == "Information Technology"
        assert all(0 <= n["finance_score"] <= 1 for n in screen["names"])

        cached = core_sel.run_screen(Config())
        assert cached["from_cache"] is True
        assert calls["n"] == 1

        assert list(tmp_path.glob("selection_screen_*.json"))


class TestRunSelectionOrchestrator:
    def _patch_screen(self, monkeypatch):
        monkeypatch.setattr(core_sel, "run_screen", lambda cfg, force_refresh=False: {**_screen(), "from_cache": True})

    def _briefing(self):
        return MarketBriefing(
            regime_summary="Risk-on with AI capex leading",
            macro_risks=["Fed holds rates"],
            trending_themes=["AI datacenters"],
            overall_market_risk=0.3,
            sectors=[
                SectorView(sector="Information Technology", trend_score=0.9, regime="accelerating",
                           driver="AI capex", fragility=0.2, conviction=0.8),
                SectorView(sector="Financials", trend_score=0.4, regime="sustained",
                           driver="Rate cuts", fragility=0.5, conviction=0.6),
                SectorView(sector="Energy", trend_score=0.3, regime="fading",
                           driver="Oil soft", fragility=0.9, conviction=0.4),
                SectorView(sector="Utilities", trend_score=0.2, regime="fading",
                           driver="Defensive demand", fragility=0.4, conviction=0.5),
            ],
        )

    def _verdicts(self):
        return {
            "A1": SelectionVerdict(include=True, conviction=0.9, thesis="t1", catalysts=[], overhangs=[], earnings_in_days=20),
            "A2": SelectionVerdict(include=False, conviction=0.2, thesis="veto", catalysts=[], overhangs=["dilution"], earnings_in_days=None),
            "B1": SelectionVerdict(include=True, conviction=0.6, thesis="t2", catalysts=[], overhangs=[], earnings_in_days=3),
            "C1": SelectionVerdict(include=True, conviction=0.7, thesis="t3", catalysts=[], overhangs=[], earnings_in_days=None),
            "C2": SelectionVerdict(include=True, conviction=0.5, thesis="t4", catalysts=[], overhangs=[], earnings_in_days=None),
        }

    def test_stage_screen_no_llm(self, monkeypatch):
        self._patch_screen(monkeypatch)
        out = sel_mod.run_selection(Config(), stage="screen")
        assert out["briefing"] is None
        assert out["selection"] == []
        assert out["degraded"] == {}

    def test_full_blend_gates_and_veto(self, monkeypatch):
        self._patch_screen(monkeypatch)
        monkeypatch.setattr(sel_mod, "run_market_briefing", lambda *a, **k: self._briefing())
        candidates = {}

        def fake_verdicts(cands, cfg, settings=None, mcp_settings=None, force_refresh=False):
            candidates["tickers"] = {c["ticker"] for c in cands}
            return self._verdicts()

        monkeypatch.setattr(sel_mod, "run_name_verdicts", fake_verdicts)
        cfg = Config(selection_top_n=10)
        out = sel_mod.run_selection(cfg, stage="all")

        assert out["degraded"] == {}
        tech = next(s for s in out["sectors"] if s["sector"] == "Information Technology")
        assert tech["final_score"] == pytest.approx(0.6 * 0.9 + 0.4 * 0.9 * (1 - 0.5 * 0.2), abs=1e-3)
        util = next(s for s in out["sectors"] if s["sector"] == "Utilities")
        assert util["llm"] is not None
        assert util["final_score"] == pytest.approx(0.6 * 0.2 + 0.4 * 0.2 * (1 - 0.5 * 0.4), abs=1e-3)

        tickers = [n["ticker"] for n in out["selection"]]
        assert tickers == ["A1", "B1", "C1", "C2", "A3"]
        assert out["vetoed"] == ["A2"]
        a1 = out["selection"][0]
        assert a1["final_score"] == pytest.approx(0.5 * 0.95 + 0.5 * 0.9, abs=1e-3)
        a3 = out["selection"][4]
        assert a3["conviction"] is None
        b1 = out["selection"][1]
        assert b1["verdict"]["event_risk"] is True
        assert "D1" not in tickers
        assert candidates["tickers"] == {"A1", "A2", "A3", "B1", "C1", "C2"}

    def test_top_n_limit(self, monkeypatch):
        self._patch_screen(monkeypatch)
        monkeypatch.setattr(sel_mod, "run_market_briefing", lambda *a, **k: self._briefing())
        monkeypatch.setattr(sel_mod, "run_name_verdicts", lambda *a, **k: self._verdicts())
        out = sel_mod.run_selection(Config(selection_top_n=8), stage="all")
        assert len(out["selection"]) <= 8

    def test_briefing_failure_degrades(self, monkeypatch):
        self._patch_screen(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("mcp down")

        monkeypatch.setattr(sel_mod, "run_market_briefing", boom)
        monkeypatch.setattr(sel_mod, "run_name_verdicts", boom)
        out = sel_mod.run_selection(Config(), stage="all")
        assert "briefing" in out["degraded"]
        assert "verdicts" in out["degraded"]
        assert out["briefing"] is None
        tech = next(s for s in out["sectors"] if s["sector"] == "Information Technology")
        assert tech["final_score"] == pytest.approx(tech["quant_score"])
        assert len(out["selection"]) > 0

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError):
            sel_mod.run_selection(Config(), stage="warp")
