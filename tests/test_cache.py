"""Tests for core/data/cache.py (uses tmp dir, no network)."""

import pandas as pd
import pytest
from core.data import cache as cache_mod
from core.data.cache import cache_path, latest_cache_file, read_cache, read_prices, write_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
    return tmp_path


def _table(prices):
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "AAA": prices,
    })


def test_cache_path_format(cache_dir):
    assert cache_path("AAA").startswith(f"{cache_dir}/AAA_")
    assert cache_path("AAA").endswith(".csv")


def test_read_cache_miss(cache_dir):
    assert read_cache("AAA") is None


def test_write_and_read_roundtrip(cache_dir):
    table = _table([10.0, 11.0, 12.0])
    write_cache("AAA", table)
    cached = read_cache("AAA")
    assert list(cached["AAA"]) == [10.0, 11.0, 12.0]
    assert len(cached) == 3


def test_latest_cache_file_missing(cache_dir):
    assert latest_cache_file("ZZZ") is None


def test_latest_cache_file_picks_most_recent(cache_dir):
    write_cache("AAA", _table([1.0, 2.0, 3.0]))
    path = latest_cache_file("AAA")
    assert path is not None
    assert path.endswith(".csv")


def test_read_prices(cache_dir):
    write_cache("AAA", _table([10.0, 11.0, 12.0]))
    series = read_prices("AAA")
    assert list(series) == [10.0, 11.0, 12.0]


def test_read_prices_missing(cache_dir):
    assert read_prices("ZZZ") is None
