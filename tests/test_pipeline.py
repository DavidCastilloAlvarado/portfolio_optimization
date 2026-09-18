"""Tests for core/pipeline.py (data layer mocked)."""

import numpy as np
import pandas as pd
import pytest
from config.defaults import Config
from core import pipeline


def _raw_data(n_days=60):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    return pd.DataFrame({
        "Date": dates,
        "AAA": 100 * (1.01) ** np.arange(n_days),
        "BBB": 100 * (1.005) ** np.arange(n_days),
    })


@pytest.fixture
def mock_data(monkeypatch):
    monkeypatch.setattr(pipeline, "bulk_stocks", lambda shares, days: _raw_data())
    monkeypatch.setattr(pipeline, "run_backtest", lambda *a, **k: None)


class TestRunPipeline:
    def test_returns_full_result(self, mock_data):
        cfg = Config(shares=["AAA", "BBB"], days=60, w_limits=(0.02, 0.9), monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert result["names"] == ["AAA", "BBB"]
        assert len(result["weights"]) == 2
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert result["mean"] > 0
        assert result["std"] > 0
        assert "Mean-Variance" in result["strategy"]
        assert result["records"] == 60

    def test_min_variance_strategy(self, mock_data):
        cfg = Config(shares=["AAA", "BBB"], w_limits=(0.02, 0.9), min_variance=True, monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert "Minimal Variance" in result["strategy"]

    def test_kelly_growth_strategy(self, mock_data):
        cfg = Config(shares=["AAA", "BBB"], w_limits=(0.02, 0.9), strategy="kelly", kelly_fraction=0.5, monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert "Growth" in result["strategy"]
        assert "Kelly" in result["strategy"]
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        assert result["mean"] > 0

    def test_llm_bl_strategy(self, mock_data, monkeypatch):
        def fake_run_blm(returns, names, cov, cfg):
            assert list(names) == ["AAA", "BBB"]
            return {
                "posterior_mean": np.array([0.0015, 0.0005]),
                "prior_mean": np.array([0.0007, 0.0003]),
                "views": {
                    "AAA": {"q": 0.0015, "omega": 1e-6, "samples": [0.0015, 0.0015], "n_ok": 2},
                    "BBB": {"q": 0.0005, "omega": 1e-6, "samples": [0.0005], "n_ok": 1},
                },
                "view_names": ["AAA", "BBB"],
                "from_cache": False,
                "degraded": False,
                "model": "test-model",
            }

        monkeypatch.setattr(pipeline, "run_blm", fake_run_blm)
        cfg = Config(shares=["AAA", "BBB"], w_limits=(0.02, 0.9), strategy="llm_bl",
                     llm_samples=3, monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert "Black-Litterman" in result["strategy"]
        assert "LLM views" in result["strategy"]
        assert abs(sum(result["weights"]) - 1.0) < 1e-6
        lv = result["llm_views"]
        assert lv["model"] == "test-model"
        assert lv["samples_per_asset"] == 3
        assert [a["ticker"] for a in lv["assets"]] == ["AAA", "BBB"]
        assert lv["assets"][0]["predicted_daily_return_pct"] == pytest.approx(0.15)
        assert all(a["samples_ok"] >= 1 for a in lv["assets"])

    def test_llm_bl_degraded_strategy(self, mock_data, monkeypatch):
        def fake_run_blm(returns, names, cov, cfg):
            prior = np.array([0.0007, 0.0003])
            return {
                "posterior_mean": prior,
                "prior_mean": prior,
                "views": {},
                "view_names": [],
                "from_cache": False,
                "degraded": True,
                "model": "test-model",
            }

        monkeypatch.setattr(pipeline, "run_blm", fake_run_blm)
        cfg = Config(shares=["AAA", "BBB"], w_limits=(0.02, 0.9), strategy="llm_bl", monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert "Black-Litterman" in result["strategy"]
        assert "equilibrium fallback" in result["strategy"]
        assert result["llm_views"]["degraded"] is True
        assert result["llm_views"]["assets"] == []

    def test_per_ticker_bounds_applied(self, mock_data):
        cfg = Config(
            shares=["AAA", "BBB"],
            w_limits=(0.02, 0.9),
            w_limits_per_ticker={"AAA": (0.6, 0.9)},
            monthly_delta=0.0,
        )
        result = pipeline.run_pipeline(cfg)
        weights = dict(zip(result["names"], result["weights"]))
        assert 0.6 - 1e-6 <= weights["AAA"] <= 0.9 + 1e-6

    def test_backtest_attached_when_present(self, mock_data, monkeypatch):
        monkeypatch.setattr(pipeline, "run_backtest", lambda *a, **k: {"lump_sum": {}})
        cfg = Config(shares=["AAA", "BBB"], w_limits=(0.02, 0.9), monthly_delta=0.0)
        result = pipeline.run_pipeline(cfg)
        assert result["backtest"] == {"lump_sum": {}}
