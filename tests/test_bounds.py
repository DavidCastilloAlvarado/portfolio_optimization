"""Tests for core/optimization/bounds.py."""

from config.defaults import Config
from core.optimization import build_asset_bounds


def test_global_fallback():
    cfg = Config(shares=["A", "B"], w_limits=(0.02, 0.12))
    assert build_asset_bounds(["A", "B"], cfg) == [(0.02, 0.12), (0.02, 0.12)]


def test_per_ticker_overrides():
    cfg = Config(
        shares=["A", "B"],
        w_limits=(0.02, 0.12),
        w_limits_per_ticker={"A": (0.1, 0.4)},
    )
    assert build_asset_bounds(["A", "B"], cfg) == [(0.1, 0.4), (0.02, 0.12)]


def test_missing_side_falls_back_to_global():
    cfg = Config(
        shares=["A", "B"],
        w_limits=(0.02, 0.12),
        w_limits_per_ticker={"A": (0.1, None), "B": (None, 0.3)},
    )
    assert build_asset_bounds(["A", "B"], cfg) == [(0.1, 0.12), (0.02, 0.3)]


def test_case_insensitive_ticker_match():
    cfg = Config(
        shares=["a", "b"],
        w_limits=(0.02, 0.12),
        w_limits_per_ticker={"A": (0.05, 0.5)},
    )
    assert build_asset_bounds(["a", "b"], cfg) == [(0.05, 0.5), (0.02, 0.12)]


def test_unknown_ticker_uses_global():
    cfg = Config(
        shares=["A", "C"],
        w_limits=(0.02, 0.12),
        w_limits_per_ticker={"A": (0.1, 0.4)},
    )
    assert build_asset_bounds(["A", "C"], cfg) == [(0.1, 0.4), (0.02, 0.12)]
