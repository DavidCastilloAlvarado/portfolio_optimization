"""Tests for core/optimization/solvers.py."""

import numpy as np
import pytest
from core.optimization import optimize, solve_kelly_growth, solve_mean_variance, solve_min_variance


class TestSolveMeanVariance:
    def test_basic_optimization(self, mean_ret, cov_ret):
        """Should return valid weights that sum to 1."""
        w = solve_mean_variance(mean_ret, cov_ret, 0.0001, (0.0, 1.0))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)

    def test_constrained_weights(self, mean_ret, cov_ret):
        """Should respect weight limits."""
        w = solve_mean_variance(mean_ret, cov_ret, 0.0001, (0.1, 0.5))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.1 - 1e-6 <= wi <= 0.5 + 1e-6 for wi in w)

    def test_per_asset_bounds(self, mean_ret, cov_ret):
        """Should respect a list of per-asset (min, max) tuples."""
        w = solve_mean_variance(mean_ret, cov_ret, 0.0001, [(0.0, 1.0), (0.1, 0.5), (0.0, 1.0)])
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert 0.1 - 1e-6 <= w[1] <= 0.5 + 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)


class TestSolveMinVariance:
    def test_basic_optimization(self, cov_ret):
        """Should return valid weights that sum to 1."""
        w = solve_min_variance(cov_ret, (0.0, 1.0), 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(wi >= -1e-6 for wi in w)

    def test_weight_upper_bound(self, cov_ret):
        """Should not exceed max weight."""
        w = solve_min_variance(cov_ret, (0.05, 0.4), 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(wi <= 0.4 + 1e-6 for wi in w)

    def test_per_asset_bounds(self, cov_ret):
        """Should respect a list of per-asset (min, max) tuples."""
        w = solve_min_variance(cov_ret, [(0.0, 0.6), (0.1, 0.3), (0.0, 1.0)], 3)
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert 0.0 <= w[0] <= 0.6 + 1e-6
        assert 0.1 - 1e-6 <= w[1] <= 0.3 + 1e-6
        assert 0.0 <= w[2] <= 1.0 + 1e-6

    def test_min_bound_enforced(self, two_cov):
        """Min bound should be applied (not hardcoded to 0)."""
        w = solve_min_variance(two_cov, (0.3, 0.8), 2)
        assert len(w) == 2
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.3 - 1e-6 <= wi <= 0.8 + 1e-6 for wi in w)
        assert w[1] == pytest.approx(0.3, abs=1e-4)


class TestSolveKellyGrowth:
    def test_basic_optimization(self, mean_ret, cov_ret):
        """Should return valid weights that sum to 1."""
        w = solve_kelly_growth(mean_ret, cov_ret, 0.5, (0.0, 1.0))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)

    def test_respects_bounds(self, mean_ret, cov_ret):
        """Should respect weight limits."""
        w = solve_kelly_growth(mean_ret, cov_ret, 0.5, (0.1, 0.5))
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert all(0.1 - 1e-6 <= wi <= 0.5 + 1e-6 for wi in w)

    def test_per_asset_bounds(self, mean_ret, cov_ret):
        """Should respect a list of per-asset (min, max) tuples."""
        w = solve_kelly_growth(mean_ret, cov_ret, 0.5, [(0.0, 1.0), (0.1, 0.5), (0.0, 1.0)])
        assert len(w) == 3
        assert abs(np.sum(w) - 1.0) < 1e-6
        assert 0.1 - 1e-6 <= w[1] <= 0.5 + 1e-6
        assert all(0.0 <= wi <= 1.0 + 1e-6 for wi in w)

    def test_favors_higher_mean(self, two_cov):
        """Higher-mean asset should receive a larger weight under growth maximization."""
        w = solve_kelly_growth(np.array([0.001, 0.004]), two_cov, 1.0, (0.0, 1.0))
        assert w[1] > w[0]

    def test_full_kelly_more_aggressive_than_half(self, mean_ret, cov_ret):
        """Full Kelly (λ=1) takes more risk (higher portfolio variance) than half-Kelly (λ=0.5)."""
        w_full = solve_kelly_growth(mean_ret, cov_ret, 1.0, (0.0, 1.0))
        w_half = solve_kelly_growth(mean_ret, cov_ret, 0.5, (0.0, 1.0))
        var_full = float(np.dot(np.dot(w_full, cov_ret), w_full))
        var_half = float(np.dot(np.dot(w_half, cov_ret), w_half))
        assert var_full >= var_half - 1e-9


class TestOptimize:
    def test_mean_variance_strategy(self, two_mean, two_cov):
        weights, mean, std, strategy = optimize(
            two_mean, two_cov, 0.0001, (0.0, 1.0), min_variance=False,
        )
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert std > 0
        assert "Mean-Variance" in strategy

    def test_min_variance_strategy(self, two_mean, two_cov):
        weights, mean, std, strategy = optimize(
            two_mean, two_cov, 0.0001, (0.0, 1.0), min_variance=True,
        )
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert std > 0
        assert "Minimal Variance" in strategy

    def test_kelly_growth_strategy(self, two_mean, two_cov):
        weights, mean, std, strategy = optimize(
            two_mean, two_cov, 0.0001, (0.0, 1.0), strategy="kelly", kelly_fraction=0.5,
        )
        assert len(weights) == 2
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert std > 0
        assert "Growth" in strategy
        assert "Kelly" in strategy

    def test_kelly_strategy_overrides_min_variance(self, two_mean, two_cov):
        """strategy=kelly must win over the legacy min_variance flag."""
        _, _, _, strategy = optimize(
            two_mean, two_cov, 0.0001, (0.0, 1.0),
            min_variance=True, strategy="kelly", kelly_fraction=0.5,
        )
        assert "Kelly" in strategy

    def test_returns_consistent_values(self, two_mean, two_cov):
        """Mean and std should match manual calculation."""
        weights, port_mean_val, port_std, _ = optimize(
            two_mean, two_cov, 0.0001, (0.0, 1.0), min_variance=False,
        )
        expected_mean = float(np.sum(weights * two_mean))
        assert port_mean_val == pytest.approx(expected_mean)
        expected_std = np.sqrt(float(np.dot(np.dot(weights, two_cov), weights)))
        assert port_std == pytest.approx(expected_std)
