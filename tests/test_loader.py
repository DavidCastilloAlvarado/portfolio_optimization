"""Tests for core/data/loader.py (network mocked)."""

import pandas as pd
import pytest
from core.data import loader as loader_mod
from core.data.cache import read_cache
from core.data.loader import bulk_stocks, fetch_prices, get_unix_time, load_table, prepare_returns


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _yahoo_payload(closes):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [1767225600, 1767312000, 1767398400],
                    "indicators": {"quote": [{"close": closes}]},
                }
            ]
        }
    }


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_mod, "read_cache", lambda ticker: None)
    monkeypatch.setattr(loader_mod, "write_cache", lambda ticker, table: None)
    return tmp_path


class TestGetUnixTime:
    def test_looks_back_in_days(self):
        init_time, end_time = get_unix_time(10)
        assert end_time > init_time
        assert end_time - init_time == 10 * 86400


class TestFetchPrices:
    def test_parses_closes(self, monkeypatch):
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse(_yahoo_payload([10.0, 11.0, 12.0])))
        table = fetch_prices("AAA", 0, 1)
        assert list(table["AAA"]) == [10.0, 11.0, 12.0]
        assert "Date" in table.columns

    def test_error_payload_raises(self, monkeypatch):
        monkeypatch.setattr(
            loader_mod.requests, "get",
            lambda *a, **k: FakeResponse({"chart": {"result": [None], "error": "No data found, symbol not found"}}),
        )
        with pytest.raises(ValueError, match="No data found"):
            fetch_prices("NOPE", 0, 1)

    def test_empty_timestamps_raises(self, monkeypatch):
        payload = {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{"close": []}]}}]}}
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse(payload))
        with pytest.raises(ValueError, match="No price data"):
            fetch_prices("AAA", 0, 1)


class TestLoadTable:
    def test_uses_cache_when_present(self, monkeypatch):
        cached = pd.DataFrame({"Date": [1], "AAA": [1.0]})
        monkeypatch.setattr(loader_mod, "read_cache", lambda ticker: cached)
        table = load_table("AAA", 0, 1)
        assert table is cached


class TestBulkStocks:
    def test_merges_tickers(self, cache_dir, monkeypatch):
        def fake_load(name, init, end):
            return pd.DataFrame({
                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                name: [10.0, 11.0] if name == "AAA" else [5.0, 5.5],
            })

        monkeypatch.setattr(loader_mod, "load_table", fake_load)
        data = bulk_stocks(["AAA", "BBB"], 30)
        assert list(data.columns) == ["Date", "AAA", "BBB"]
        assert len(data) == 2


class TestPrepareReturns:
    def _data(self):
        return pd.DataFrame({
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "AAA": [100.0, 110.0, 121.0],
            "BBB": [50.0, 50.0, 55.0],
        })

    def test_daily_returns(self):
        data, returns = prepare_returns(self._data(), resample=None)
        assert data.index.name == "Date"
        # descending: iloc[0] is 01-03, iloc[1] is 01-02
        assert returns["AAA"].iloc[0] == pytest.approx((121.0 - 110.0) / 110.0)
        assert returns["AAA"].iloc[1] == pytest.approx((110.0 - 100.0) / 100.0)
        assert returns["BBB"].iloc[0] == pytest.approx((55.0 - 50.0) / 50.0)
        assert returns["BBB"].iloc[1] == pytest.approx(0.0)

    def test_weekly_resample(self):
        data, returns = prepare_returns(self._data(), resample="week")
        assert "step" not in data.columns
        assert len(data) <= 3

    def test_monthly_resample(self):
        data, returns = prepare_returns(self._data(), resample="month")
        assert len(data) == 1
