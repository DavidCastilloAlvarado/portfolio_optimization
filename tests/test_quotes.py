"""Tests for core/data/quotes.py (Yahoo crumb session + batch quotes, HTTP mocked)."""

import pytest

from core.data import quotes as q


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload if payload is not None else {}
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload=None, crumb="test-crumb"):
        self.crumb = crumb
        self.payload = payload or {"quoteResponse": {"result": []}}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.payload)


def _quote(symbol, **overrides):
    qd = {"symbol": symbol, "marketCap": 1e12, "regularMarketPrice": 100.0,
          "trailingPE": 30.0, "forwardPE": None, "epsTrailingTwelveMonths": 4.0}
    qd.update(overrides)
    return qd


class TestGetCrumbSession:
    def test_cookie_then_crumb(self, monkeypatch):
        import requests as req
        seen = []

        class S(req.Session):
            def get(self, url, timeout=None):
                seen.append(url)
                return FakeResponse(text="the-crumb") if "getcrumb" in url else FakeResponse()

        monkeypatch.setattr(req, "Session", S)
        session = q.get_crumb_session()
        assert seen[0] == "https://fc.yahoo.com"
        assert "getcrumb" in seen[1]
        assert session.crumb is not None
        assert session.headers.get("x-requested-with") == "XMLHttpRequest"

    def test_crumb_text_used_verbatim(self, monkeypatch):
        import requests as req
        calls = {"n": 0}

        class S(req.Session):
            def get(self, url, timeout=None):
                calls["n"] += 1
                return FakeResponse(text="CR-123" if calls["n"] == 2 else "")

        monkeypatch.setattr(req, "Session", S)
        session = q.get_crumb_session()
        assert session.crumb == "CR-123"


class TestFetchBatchQuotes:
    def test_single_chunk_filters_nulls(self):
        session = FakeSession(payload={"quoteResponse": {"result": [
            _quote("AAA"),
            _quote("BBB", marketCap=None),
            {"symbol": None, "marketCap": 5},
            _quote("CCC", netIncomeToCommon=1e10, unknown_field=1),
        ]}})
        out = q.fetch_batch_quotes(["AAA", "BBB", "CCC"], session=session)
        assert set(out) == {"AAA", "BBB", "CCC"}
        assert out["AAA"]["marketCap"] == 1e12
        assert "marketCap" not in out["BBB"]
        assert "unknown_field" not in out["CCC"]
        assert out["CCC"]["netIncomeToCommon"] == 1e10
        assert len(session.calls) == 1
        assert session.calls[0][1]["crumb"] == "test-crumb"
        assert "marketCap" in session.calls[0][1]["fields"]

    def test_chunking_over_60(self, monkeypatch):
        import time as time_mod
        monkeypatch.setattr(time_mod, "sleep", lambda s: None)
        session = FakeSession()
        symbols = [f"T{i:02d}" for i in range(130)]
        out = q.fetch_batch_quotes(symbols, session=session)
        assert len(session.calls) == 3
        first = session.calls[0][1]["symbols"]
        assert first.count(",") == 59
        last = session.calls[2][1]["symbols"]
        assert last.count(",") == 9
        assert len(out) == 0

    def test_http_error_propagates(self):
        class Boom(FakeResponse):
            def raise_for_status(self):
                raise RuntimeError("401")

        class S(FakeSession):
            def get(self, url, params=None, timeout=None):
                self.calls.append((url, params))
                return Boom({})

        session = S()
        with pytest.raises(RuntimeError, match="401"):
            q.fetch_batch_quotes(["AAA"], session=session)

    def test_no_session_arg_uses_crumb_session(self, monkeypatch):
        made = {"n": 0}
        fake = FakeSession()

        def fake_crumb():
            made["n"] += 1
            return fake

        monkeypatch.setattr(q, "get_crumb_session", fake_crumb)
        out = q.fetch_batch_quotes(["AAA"])
        assert made["n"] == 1
        assert out == {}
