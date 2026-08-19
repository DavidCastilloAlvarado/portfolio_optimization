"""Tests for core/data/loader.py (network mocked)."""

import pandas as pd
import pytest
from core.data import loader as loader_mod
from core.data import cache as cache_mod
from core.data.cache import read_cache
from core.data.loader import (
    bulk_stocks,
    fetch_prices,
    fetch_prices_justetf,
    get_unix_time,
    is_isin,
    load_table,
    prepare_returns,
)


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

    def test_normalizes_yahoo_intraday_timestamps(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "read_cache", lambda ticker: None)
        monkeypatch.setattr(loader_mod, "write_cache", lambda ticker, table: None)
        monkeypatch.setattr(loader_mod, "fetch_prices", lambda name, init, end: pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01 13:30:00", "2024-01-02 13:30:00"], utc=True),
            "AAA": [10.0, 11.0],
        }))
        table = load_table("AAA", 0, 1)
        assert table["Date"].tolist() == pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True).tolist()


def _justetf_payload(closes):
    return {
        "latestQuote": {"raw": closes[-1], "localized": "1.0"},
        "series": [
            {"date": d, "value": {"raw": v, "localized": "1.0"}}
            for d, v in zip(["2024-01-01", "2024-01-02", "2024-01-03"], closes)
        ],
    }


class TestIsIsin:
    def test_valid_isin(self):
        assert is_isin("IE00BFMXXD54")

    def test_lowercase_isin_matches(self):
        assert is_isin("ie00bfmxxd54")

    def test_tickers_are_not_isins(self):
        assert not is_isin("AAPL")
        assert not is_isin("BRK.B")
        assert not is_isin("SPY")

    def test_invalid_shapes_rejected(self):
        assert not is_isin("IE00BFMXXD5")
        assert not is_isin("IE00BFMXXD545")
        assert not is_isin("1E00BFMXXD54")
        assert not is_isin("IE00BFMXXD5A")


class TestFetchPricesJustetf:
    def test_parses_series(self, monkeypatch):
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse(_justetf_payload([90.05, 89.19, 88.48])))
        table = fetch_prices_justetf("IE00BFMXXD54", 1704067200, 1704931200)
        assert list(table["IE00BFMXXD54"]) == [90.05, 89.19, 88.48]
        assert table["Date"].tolist() == pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True).tolist()
        assert str(table["Date"].dtype) == "datetime64[ns, UTC]"

    def test_url_contains_isin_and_date_range(self, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            return FakeResponse(_justetf_payload([1.0]))

        monkeypatch.setattr(loader_mod.requests, "get", fake_get)
        fetch_prices_justetf("IE00BFMXXD54", 1704067200, 1704931200)
        assert "/api/etfs/IE00BFMXXD54/performance-chart" in captured["url"]
        assert "dateFrom=2024-01-01" in captured["url"]
        assert "dateTo=2024-01-11" in captured["url"]

    def test_empty_series_raises(self, monkeypatch):
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse({"series": []}))
        with pytest.raises(ValueError, match="No price data"):
            fetch_prices_justetf("IE00BFMXXD54", 0, 1)


class TestLoadTableFallback:
    def _no_cache(self, monkeypatch):
        monkeypatch.setattr(loader_mod, "read_cache", lambda ticker: None)
        monkeypatch.setattr(loader_mod, "write_cache", lambda ticker, table: None)

    def test_falls_back_to_justetf_for_isin(self, monkeypatch):
        self._no_cache(monkeypatch)
        monkeypatch.setattr(loader_mod, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(ValueError("Yahoo down")))
        table = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "IE00BFMXXD54": [90.05]})
        monkeypatch.setattr(loader_mod, "fetch_prices_justetf", lambda name, init, end: table)
        assert load_table("IE00BFMXXD54", 0, 1) is table

    def test_no_fallback_for_plain_ticker(self, monkeypatch):
        self._no_cache(monkeypatch)
        def no_justetf(*a, **k):
            raise AssertionError("JustETF should not be called")
        monkeypatch.setattr(loader_mod, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(ValueError("Yahoo down")))
        monkeypatch.setattr(loader_mod, "fetch_prices_justetf", no_justetf)
        with pytest.raises(ValueError, match="Yahoo down"):
            load_table("AAPL", 0, 1)

    def test_no_fallback_when_yahoo_succeeds(self, monkeypatch):
        self._no_cache(monkeypatch)
        table = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "IE00BFMXXD54": [90.05]})
        monkeypatch.setattr(loader_mod, "fetch_prices", lambda name, init, end: table)
        def no_justetf(*a, **k):
            raise AssertionError("JustETF should not be called")
        monkeypatch.setattr(loader_mod, "fetch_prices_justetf", no_justetf)
        assert load_table("IE00BFMXXD54", 0, 1) is table


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

    def test_merges_cached_table_with_justetf_table(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))
        cache_mod.write_cache("AAA", pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01 13:30:00", "2024-01-02 13:30:00", "2024-01-03 13:30:00"], utc=True),
            "AAA": [10.0, 11.0, 12.0],
        }))

        monkeypatch.setattr(loader_mod, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(ValueError("Yahoo down")))
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse(_justetf_payload([5.0, 5.5, 5.6])))

        data = bulk_stocks(["AAA", "IE00BFMXXD54"], 30)
        assert list(data.columns) == ["Date", "AAA", "IE00BFMXXD54"]
        assert len(data) == 3

    def test_merges_fresh_yahoo_and_fresh_justetf_tables(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", str(tmp_path))

        def fake_yahoo(name, init, end):
            if name == "IE00BFMXXD54":
                raise ValueError("Yahoo has no data for ISINs")
            return pd.DataFrame({
                "Date": pd.to_datetime(["2024-01-01 13:30:00", "2024-01-02 13:30:00", "2024-01-03 13:30:00"], utc=True),
                "AAA": [10.0, 11.0, 12.0],
            })

        monkeypatch.setattr(loader_mod, "fetch_prices", fake_yahoo)
        monkeypatch.setattr(loader_mod.requests, "get", lambda *a, **k: FakeResponse(_justetf_payload([5.0, 5.5, 5.6])))

        data = bulk_stocks(["AAA", "IE00BFMXXD54"], 30)
        assert list(data.columns) == ["Date", "AAA", "IE00BFMXXD54"]
        assert len(data) == 3
        assert list(data["Date"]) == pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True).tolist()


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
