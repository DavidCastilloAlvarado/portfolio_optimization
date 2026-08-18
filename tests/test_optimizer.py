"""Tests for optimizer.py."""

import numpy as np
import pytest
from optimizer import port_mean, port_var, port_mean_var, optimize, solve_mean_variance, solve_min_variance


# ── port_mean / port_var / port_mean_var ─────────────────────────────

class TestPortMean:
    def test_equal_weights(self):
        W = np.array([0.5, 0.5])
        R = np.array([0.01, 0.02])
        assert port_mean(W, R) == pytest.approx(0.015)

    def test_single_asset(self):
        W = np.array([1.0])
        R = np.array([0.05])
        assert port_mean(W, R) == pytest.approx(0.05)

    def test_zero_weights(self):
        W = np.array([0.0, 0.0])
        R = np.array([0.01, 0.02])
        assert port_mean(W, R) == pytest.approx(0.0)


class TestPortVar:
    def test_no_covariance(self):
        W = np.array([0.5, 0.5])
        C = np.array([[0.04, 0.0], [0.0, 0.09]])
        # var = 0.5^2 * 0.04 + 0.5^2 * 0.09 = 0.01 + 0.0225 = 0.0325
        assert port_var(W, C) == pytest.approx(0.0325)

    def test_single_asset(self):
        W = np.array([1.0])
        C = np.array([[0.04]])
        assert port_var(W, C) == pytest.approx(0.04)

    def test_perfect_correlation(self):
        W = np.array([0.5, 0.5])
        C = np.array([[0.04, 0.04], [0.04, 0.04]])
        assert port_var(W, C) == pytest.approx(0.04)


class TestPortMeanVar:
    def test_returns_mean_and_var(self):
        W = np.array([0.5, 0.5])
        R = np.array([0.01, 0.03])
        C = np.array([[0.01, 0.0], [0.0, 0.04]])
        m, v = port_mean_var(W, R, C)
        assert m == pytest.approx(0.02)
        assert v == pytest.approx(0.0125)


# ── solve_mean_variance ──────────────────────────────────────────────

class TestSolveMeanVariance:
    def test_basic_optimization(self):
        """Should return valid weights that sum to 1."""
        mean_ret = np.array([0.001, 0.002, 0.0015])
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        rf = 0.0001
        w = solve_mean_variance(mean_ret, cov_ret, rf, (0.0, 1.0))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)

    def test_constrained_weights(self):
        """Should respect weight limits."""
        mean_ret = np.array([0.001, 0.002, 0.0015])
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        rf = 0.0001
        w = solve_mean_variance(mean_ret, cov_ret, rf, (0.1, 0.5))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.1 - 1e-6 <= wi <= 0.5 + 1e-6 for wi in w)

    def test_per_asset_bounds(self):
        """Should respect a list of per-asset (min, max) tuples."""
        mean_ret = np.array([0.001, 0.002, 0.0015])
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        rf = 0.0001
        w = solve_mean_variance(mean_ret, cov_ret, rf, [(0.0, 1.0), (0.1, 0.5), (0.0, 1.0)])
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert 0.1 - 1e-6 <= w[1] <= 0.5 + 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)


# ── solve_min_variance ──────────────────────────────────────────────

class TestSolveMinVariance:
    def test_basic_optimization(self):
        """Should return valid weights that sum to 1."""
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        w = solve_min_variance(cov_ret, (0.0, 1.0), 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(wi >= -1e-6 for wi in w)

    def test_weight_upper_bound(self):
        """Should not exceed max weight."""
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        w = solve_min_variance(cov_ret, (0.05, 0.4), 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(wi <= 0.4 + 1e-6 for wi in w)

    def test_per_asset_bounds(self):
        """Should respect a list of per-asset (min, max) tuples."""
        cov_ret = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0009, 0.0002],
            [0.00005, 0.0002, 0.0006],
        ])
        w = solve_min_variance(cov_ret, [(0.0, 0.6), (0.1, 0.3), (0.0, 1.0)], 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert 0.0 <= w[0] <= 0.6 + 1e-6
        assert 0.1 - 1e-6 <= w[1] <= 0.3 + 1e-6
        assert 0.0 <= w[2] <= 1.0 + 1e-6

    def test_min_bound_enforced(self):
        """Min bound should be applied (not hardcoded to 0)."""
        cov_ret = np.array([
            [0.0004, 0.0001],
            [0.0001, 0.0009],
        ])
        w = solve_min_variance(cov_ret, (0.3, 0.8), 2)
        assert len(w) == 2
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.3 - 1e-6 <= wi <= 0.8 + 1e-6 for wi in w)
        assert w[1] == pytest.approx(0.3, abs=1e-4)


# ── optimize (wrapper) ──────────────────────────────────────────────

class TestOptimize:
    def test_mean_variance_strategy(self):
        mean_ret = np.array([0.001, 0.002])
        cov_ret = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        rf = 0.0001
        weights, mean, std, strategy = optimize(
            mean_ret, cov_ret, rf, (0.0, 1.0), min_variance=False,
        )
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert std > 0
        assert "Mean-Variance" in strategy

    def test_min_variance_strategy(self):
        mean_ret = np.array([0.001, 0.002])
        cov_ret = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
        rf = 0.0001
        weights, mean, std, strategy = optimize(
            mean_ret, cov_ret, rf, (0.0, 1.0), min_variance=True,
        )
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert std > 0
        assert "Minimal Variance" in strategy

    def test_returns_consistent_values(self):
        """Mean and std should match manual calculation."""
        mean_ret = np.array([0.001, 0.002])
        cov_ret = np.array([[0.0004, 0.0], [0.0, 0.0009]])
        rf = 0.0001
        weights, port_mean_val, port_std, _ = optimize(
            mean_ret, cov_ret, rf, (0.0, 1.0), min_variance=False,
        )
        # Manual check: mean = sum(weights * mean_ret)
        expected_mean = float(np.sum(weights * mean_ret))
        assert port_mean_val == pytest.approx(expected_mean)
        # Manual check: std = sqrt(weights^T * cov * weights)
        expected_std = np.sqrt(float(np.dot(np.dot(weights, cov_ret), weights)))
        assert port_std == pytest.approx(expected_std)


# ── Config ──────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        from config.defaults import Config
        cfg = Config()
        assert cfg.resample == "none"
        assert cfg.days == 720
        assert cfg.min_variance is False
        assert cfg.monto_usd == 10000.0
        assert cfg.monthly_delta == 300.0
        assert cfg.sim_days == 252
        assert cfg.risk_free_annual_perc == 5.0

    def test_risk_free_computation(self):
        from config.defaults import Config
        cfg = Config()
        # (1 + 0.05)^(1/365) - 1 ≈ 0.0001337
        expected = (1 + 5 / 100) ** (1 / 365) - 1
        assert cfg.risk_free == pytest.approx(expected)

    def test_get_resample_none(self):
        from config.defaults import Config
        cfg = Config(resample="none")
        assert cfg.get_resample() is None

    def test_get_resample_week(self):
        from config.defaults import Config
        cfg = Config(resample="week")
        assert cfg.get_resample() == "week"

    def test_from_dict_basic(self):
        from config.defaults import Config
        data = {
            "resample": "week",
            "days": "365",
            "shares": "AAPL, GOOG",
            "w_limits": "0.05,0.20",
            "min_variance": "true",
            "monto_usd": "5000",
            "monthly_delta": "500",
            "sim_days": "126",
            "risk_free_annual_perc": "3.5",
        }
        cfg = Config.from_dict(data)
        assert cfg.resample == "week"
        assert cfg.days == 365
        assert cfg.shares == ["AAPL", "GOOG"]
        assert cfg.w_limits == (0.05, 0.20)
        assert cfg.min_variance is True
        assert cfg.monto_usd == 5000.0
        assert cfg.monthly_delta == 500.0
        assert cfg.sim_days == 126
        assert cfg.risk_free_annual_perc == 3.5

    def test_from_dict_empty(self):
        from config.defaults import Config
        cfg = Config.from_dict({})
        assert cfg.shares == []
        assert cfg.w_limits == (0.02, 0.12)
        assert cfg.w_limits_per_ticker == {}

    def test_from_dict_per_ticker(self):
        from config.defaults import Config
        data = {
            "shares": "QQQM,SCHD",
            "w_limits": "0.02,0.12",
            "w_limits_per_ticker": "qqqm:0.02,0.12;SCHD:0.05,",
        }
        cfg = Config.from_dict(data)
        assert cfg.w_limits_per_ticker == {"QQQM": (0.02, 0.12), "SCHD": (0.05, None)}
