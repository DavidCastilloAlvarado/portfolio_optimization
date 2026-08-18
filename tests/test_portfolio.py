"""Tests for core/optimization/portfolio.py."""

import numpy as np
import pytest
from core.optimization import port_mean, port_var, port_mean_var


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
