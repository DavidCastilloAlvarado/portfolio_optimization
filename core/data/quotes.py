"""Yahoo batch quotes (market cap, P/E, growth, margins) via the cookie+crumb flow.

The v7 quote endpoint requires a session cookie plus a crumb token; the plain
v8 chart endpoint (core/data/loader.py) does not. This module keeps that
handshake in one place and returns a per-symbol dict of fundamentals.
"""

import time

import requests

from core.progress import progress_log

QUOTE_FIELDS = (
    "marketCap",
    "regularMarketPrice",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "trailingPE",
    "forwardPE",
    "epsTrailingTwelveMonths",
    "trailingAnnualRevenueGrowth",
    "profitMargins",
    "netIncomeToCommon",
    "averageDailyVolume3month",
)


def get_crumb_session() -> requests.Session:
    """A requests session carrying the Yahoo cookie + crumb token."""
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    session.get("https://fc.yahoo.com", timeout=15)
    crumb = session.get(
        "https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=15
    ).text
    session.headers["x-requested-with"] = "XMLHttpRequest"
    session.crumb = crumb
    return session


def fetch_fundamentals(symbols: list[str], session: requests.Session | None = None) -> dict[str, dict]:
    """Per-symbol fundamentals via v10 quoteSummary (financialData module).

    The v7 quote endpoint returns null for growth/margins on most tickers; the
    v10 module is the reliable source. Returns
        {SYMBOL: {profitMargins, trailingAnnualRevenueGrowth, earningsGrowth}}
    with only non-null fields; per-symbol failures are skipped.
    """
    session = session or get_crumb_session()
    out: dict[str, dict] = {}
    for index, sym in enumerate(symbols, start=1):
        progress_log("fundamentals", f"fetching {sym} ({index}/{len(symbols)})")
        try:
            response = session.get(
                f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}",
                params={"modules": "financialData", "crumb": session.crumb},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json().get("quoteSummary", {}).get("result") or []
            if not result:
                continue
            fd = result[0].get("financialData", {})

            def raw(key: str):
                value = fd.get(key)
                return value.get("raw") if isinstance(value, dict) else value

            row = {
                "profitMargins": raw("profitMargins"),
                "trailingAnnualRevenueGrowth": raw("revenueGrowth"),
                "earningsGrowth": raw("earningsGrowth"),
            }
            row = {k: v for k, v in row.items() if v is not None}
            if row:
                out[sym] = row
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
        time.sleep(0.15)
    return out


def fetch_batch_quotes(symbols: list[str], session: requests.Session | None = None) -> dict[str, dict]:
    """Fetch quote fundamentals for a list of symbols in chunks of 60.

    Returns:
        {SYMBOL: {field: value, ...}} — only fields Yahoo returned non-null.
    """
    session = session or get_crumb_session()
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 60):
        batch = symbols[i:i + 60]
        response = session.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={
                "symbols": ",".join(batch),
                "crumb": session.crumb,
                "fields": ",".join(QUOTE_FIELDS),
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json().get("quoteResponse", {}).get("result", [])
        for quote in result:
            symbol = quote.get("symbol")
            if not symbol:
                continue
            out[symbol] = {k: v for k, v in quote.items() if k in QUOTE_FIELDS and v is not None}
        time.sleep(0.3)
    return out
