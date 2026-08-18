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
