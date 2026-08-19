"""Tests for core/simulation/backtest.py (synthetic prices)."""

import numpy as np
import pandas as pd
import pytest
from core.data import cache as cache_mod
from core.simulation import run_backtest


def make_prices(n_days=60, base=100.0, growth_a=0.01, growth_b=0.02):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    a = base * (1 + growth_a) ** np.arange(n_days)
    b = base * (1 + growth_b) ** np.arange(n_days)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


def static_loader(n_days):
    return lambda shares, n: make_prices(n_days).tail(n)


class TestRunBacktest:
    def test_lump_sum_final_value(self):
        n_days = 30
        result = run_backtest(
            10000, ["A", "B"], [0.5, 0.5], n_days,
            risk_free_annual=0.0, monthly_delta=0.0,
            price_loader=static_loader(n_days),
        )
        expected = 10000 * (1 + 0.015) ** (n_days - 1)
        assert result["lump_sum"]["final_value"] == pytest.approx(expected, rel=1e-9)
        assert result["lump_sum"]["total_return_pct"] == pytest.approx(expected / 10000 * 100 - 100, rel=1e-9)
        assert result["lump_sum"]["trading_days"] == n_days - 1
        assert "dca" not in result

    def test_lump_sum_no_drawdown_for_rising_prices(self):
        result = run_backtest(
            10000, ["A", "B"], [0.5, 0.5], 40,
            price_loader=static_loader(40),
        )
        assert result["lump_sum"]["max_drawdown_pct"] == pytest.approx(0.0)

    def test_constant_returns_zero_sharpe(self):
        flat = pd.DataFrame(
            {"A": [100.0] * 30, "B": [100.0] * 30},
            index=pd.bdate_range("2026-01-01", periods=30),
        )
        result = run_backtest(
            10000, ["A", "B"], [0.5, 0.5], 30,
            price_loader=lambda s, n: flat,
        )
        assert result["lump_sum"]["sharpe_annualized"] == 0.0
        assert result["lump_sum"]["final_value"] == pytest.approx(10000)

    def test_dca_contributions(self):
        n_days = 43
        monthly_delta = 300
        result = run_backtest(
            10000, ["A", "B"], [0.5, 0.5], n_days,
            monthly_delta=monthly_delta,
            price_loader=static_loader(n_days),
        )
        dca = result["dca"]
        assert dca["months_contributed"] == 2
        assert dca["total_invested"] == 10600
        r = 1.015
        expected = 10000 * r ** 42 + 300 * r ** 21 + 300
        assert dca["final_value"] == pytest.approx(expected, rel=1e-9)

    def test_not_enough_data_returns_error(self):
        result = run_backtest(
            10000, ["A", "B"], [0.5, 0.5], 30,
            price_loader=lambda s, n: pd.DataFrame(),
        )
        assert "error" in result

    def test_missing_ticker_skipped(self):
        prices = make_prices(30)
        prices = prices[["A"]]
        result = run_backtest(
            10000, ["A", "ZZZ"], [1.0, 0.0], 30,
            price_loader=lambda s, n: prices,
        )
        assert "lump_sum" in result


class TestMixedSourceCache:
    def test_mixed_source_cache_files_align(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        dates = pd.bdate_range("2026-01-01", periods=30)
        cache_mod.write_cache("A", pd.DataFrame({
            "Date": pd.to_datetime(dates, utc=True) + pd.Timedelta(hours=13, minutes=30),
            "A": 100.0 * (1.01 ** np.arange(30)),
        }))
        cache_mod.write_cache("B", pd.DataFrame({
            "Date": pd.to_datetime(dates, utc=True),
            "B": 100.0 * (1.02 ** np.arange(30)),
        }))
        result = run_backtest(10000, ["A", "B"], [0.5, 0.5], 30, monthly_delta=0.0)
        assert "lump_sum" in result
        assert result["lump_sum"]["trading_days"] == 29
