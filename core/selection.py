"""S&P 100 selection funnel — quant screen (no LLM).

Pipeline: universe (config/sp100.json) -> per-ticker price metrics (momentum,
52w-high proximity, drawdown, volatility, relative volume) + quote fundamentals
(market cap, P/E, growth, margins) -> percentile-rank scoring -> sector
aggregation (momentum + breadth + dispersion) -> near-duplicate dedup
(GOOG/GOOGL style) -> ranked screen, cached daily in temp/.

The LLM layer (market briefing + per-name verdicts) lives in models/selection.py
and models/selection_agents.py and consumes this screen.

Metrics use a 400-day price window; scores are percentile ranks across the
universe so they are scale-free and comparable day to day. Historical volume
comes from a selection-only cache file (the shared price cache stores closes
only), so relative volume is available without touching core/data/loader.py.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.defaults import Config
from core.data import cache as price_cache
from core.data.loader import get_unix_time
from core.progress import progress_log

SP100_PATH = Path(__file__).resolve().parent.parent / "config" / "sp100.json"
MIN_OBS = 63
PRICE_DAYS_BACK = 400
DEDUP_CORRELATION = 0.95
GICS_SECTORS = (
    "Information Technology", "Financials", "Health Care", "Industrials",
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Materials", "Real Estate", "Utilities",
)


def load_universe(path: str | None = None) -> dict[str, dict]:
    """Load the S&P 100 universe: {TICKER: {name, sector, sub_industry}}."""
    p = Path(path or SP100_PATH)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Universe file not found: {p}. Create it with one \"TICKER\": {{\"name\", \"sector\", \"sub_industry\"}} entry per line."
        )
    try:
        raw_map = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in universe file {p}: {exc}") from exc
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError(f"Universe file {p} must be a non-empty JSON object of ticker -> entry")
    out: dict[str, dict] = {}
    for ticker, entry in raw_map.items():
        sector = str(entry.get("sector", "")).strip()
        if sector not in GICS_SECTORS:
            raise ValueError(f"Universe file {p}: ticker {ticker} has unknown GICS sector '{sector}'")
        out[str(ticker).strip().upper()] = {
            "name": str(entry.get("name", ticker)).strip(),
            "sector": sector,
            "sub_industry": str(entry.get("sub_industry", "")).strip(),
        }
    return out


def _fetch_chart(name: str, init_time: int, end_time: int) -> pd.DataFrame:
    """Daily close + volume for one ticker from Yahoo Finance Query v8."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{name}"
        f"?period1={init_time}&period2={end_time}&interval=1d"
    )
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    result = response.json()["chart"]["result"]
    if not result or result[0] is None:
        raise ValueError(f"No chart data for '{name}'")
    result = result[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps if ts is not None]
    return pd.DataFrame({
        "Date": dates,
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })


def _selection_prices_path() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{price_cache.CACHE_DIR}/selection_prices_{today}.csv"


def load_universe_prices(universe: dict[str, dict]) -> tuple[pd.DataFrame, list[str]]:
    """Daily close + volume for the whole universe, cached once per day.

    Returns:
        (prices, failures) — prices has columns {TICKER}, {TICKER}_volume
        aligned on a shared Date index (missing days are NaN); failures is the
        list of tickers whose fetch failed and were skipped.
    """
    path = _selection_prices_path()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            expected = set(universe) | {f"{ticker}_volume" for ticker in universe}
            if "Date" in df.columns and expected.issubset(df.columns):
                df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.normalize()
                df = df.dropna(subset=["Date"])
                return df.set_index("Date").sort_index(), []
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    init_time, end_time = get_unix_time(PRICE_DAYS_BACK)
    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for index, ticker in enumerate(universe, start=1):
        progress_log("screen", f"price history {index}/{len(universe)}: {ticker}")
        try:
            table = _fetch_chart(ticker, init_time, end_time)
            table = table[table["Date"].notna()]
            table["Date"] = pd.to_datetime(table["Date"], utc=True).dt.normalize()
            frames[ticker] = table.set_index("Date")
        except (requests.RequestException, ValueError, KeyError, TypeError):
            failures.append(ticker)

    if not frames:
        raise ValueError("No price data could be fetched for the selection universe")
    all_dates = set().union(*[set(f.index) for f in frames.values()])
    index = pd.DatetimeIndex(sorted(all_dates))
    cols: dict[str, pd.Series] = {}
    for ticker, table in frames.items():
        close = table["close"].reindex(index)
        volume = table["volume"].reindex(index)
        cols[ticker] = close
        cols[f"{ticker}_volume"] = volume
    prices = pd.DataFrame(cols, index=index).sort_index()
    os.makedirs(price_cache.CACHE_DIR, exist_ok=True)
    prices.reset_index().to_csv(path, index=False)
    return prices, failures


def compute_name_metrics(close: pd.Series, volume: pd.Series | None = None) -> dict | None:
    """Price-based metrics for one ticker, or None when history is too short."""
    close = close.dropna().sort_index()
    if len(close) < MIN_OBS:
        return None

    def ret(n: int) -> float | None:
        if len(close) <= n:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - n] - 1)

    window = close.iloc[-252:]
    returns = close.pct_change().dropna()
    recent = close.iloc[-63:]
    m = {
        "ret_1m": ret(21),
        "ret_3m": ret(63),
        "ret_6m": ret(126),
        "off_high_52w": float(close.iloc[-1] / window.max() - 1),
        "max_dd_3m": float((recent / recent.cummax() - 1).min()),
        "vol_3m": float(returns.iloc[-63:].std()),
        "above_ma50": bool(close.iloc[-1] > close.iloc[-50:].mean()),
    }
    if volume is not None:
        v = volume.dropna()
        if len(v) >= 63:
            base = float(v[-63:].mean())
            if base > 0:
                m["rel_volume"] = float(v[-5:].mean() / base)
    return m


def _percentile(values: dict[str, float]) -> dict[str, float]:
    """Tie-aware percentile rank in [0, 1] per key."""
    if len(values) <= 1:
        return {sym: 0.5 for sym in values}
    series = pd.Series(values, dtype=float)
    ranks = (series.rank(method="average") - 1.0) / (len(series) - 1)
    return {sym: float(rank) for sym, rank in ranks.items()}


MOMENTUM_KEYS = ("ret_1m", "ret_3m", "ret_6m", "off_high_52w", "rel_volume")


def score_universe(
    metrics: dict[str, dict],
    quotes: dict[str, dict],
) -> dict[str, dict]:
    """Percentile-rank scoring across the universe.

    finance = 0.75 * momentum composite + 0.25 * quality composite.
    Quality requires positive earnings; names without them score 0 on quality.
    """
    by_metric: dict[str, dict[str, float]] = {}
    for key in MOMENTUM_KEYS:
        by_metric[key] = {sym: m[key] for sym, m in metrics.items() if m.get(key) is not None}
    ranks: dict[str, dict[str, float]] = {key: _percentile(vals) for key, vals in by_metric.items()}

    momentum: dict[str, float] = {}
    for sym in metrics:
        parts = [ranks[key][sym] for key in MOMENTUM_KEYS if sym in ranks[key]]
        momentum[sym] = float(np.mean(parts)) if parts else 0.5

    growth = {s: q["trailingAnnualRevenueGrowth"] for s, q in quotes.items() if q.get("trailingAnnualRevenueGrowth") is not None}
    margins = {s: q["profitMargins"] for s, q in quotes.items() if q.get("profitMargins") is not None}
    growth_r = _percentile(growth)
    margins_r = _percentile(margins)

    out: dict[str, dict] = {}
    for sym, m in metrics.items():
        quote = quotes.get(sym, {})
        has_earnings = (
            quote.get("epsTrailingTwelveMonths") not in (None, 0) and quote.get("trailingPE") not in (None, 0)
        ) or (quote.get("profitMargins") is not None and quote["profitMargins"] > 0)
        quality_parts = [growth_r[sym] for sym in (sym,) if sym in growth_r] + [margins_r[sym] for sym in (sym,) if sym in margins_r]
        quality = float(np.mean(quality_parts)) if (quality_parts and has_earnings) else 0.0
        out[sym] = {
            "momentum": momentum[sym],
            "quality": quality,
            "finance": float(0.75 * momentum[sym] + 0.25 * quality),
        }
    return out


def sector_scores(scored: dict[str, dict], universe: dict[str, dict], metrics: dict[str, dict]) -> list[dict]:
    """Per-sector aggregation: member momentum, breadth, dispersion."""
    ret_3m = _percentile({s: m["ret_3m"] for s, m in metrics.items() if m.get("ret_3m") is not None})
    sectors: list[dict] = []
    for sector in GICS_SECTORS:
        members = [s for s, e in universe.items() if e["sector"] == sector and s in scored]
        if not members:
            continue
        mom = float(np.mean([ret_3m[s] for s in members if s in ret_3m] or [0.5]))
        breadth = float(np.mean([
            1.0 for s in members
            if metrics[s].get("ret_1m") not in (None, 0) and metrics[s]["ret_1m"] > 0
            and metrics[s]["off_high_52w"] > -0.05
        ] or [0.0]))
        dispersion = float(np.std([ret_3m[s] for s in members if s in ret_3m])) if len(members) > 1 else 0.0
        sectors.append({
            "sector": sector,
            "n_members": len(members),
            "momentum": round(mom, 4),
            "breadth": round(breadth, 4),
            "dispersion": round(dispersion, 4),
            "quant_score": round(0.6 * mom + 0.4 * breadth, 4),
        })
    sectors.sort(key=lambda s: s["quant_score"], reverse=True)
    return sectors


def _dedup_near_duplicates(prices: pd.DataFrame, universe: dict[str, dict], quotes: dict[str, dict]) -> set[str]:
    """Drop the smaller twin when two same-sub-industry tickers are ~identical (GOOG/GOOGL)."""
    removed: set[str] = set()
    by_sub: dict[str, list[str]] = {}
    for sym, e in universe.items():
        if e.get("sub_industry"):
            by_sub.setdefault(e["sub_industry"], []).append(sym)
    for group in by_sub.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a in removed or b in removed or a not in prices or b not in prices:
                    continue
                ra = prices[a].pct_change().dropna().iloc[-126:]
                rb = prices[b].pct_change().dropna().iloc[-126:]
                common = ra.index.intersection(rb.index)
                if len(common) < 60:
                    continue
                corr = float(np.corrcoef(ra.loc[common], rb.loc[common])[0, 1])
                cap_a = quotes.get(a, {}).get("marketCap")
                cap_b = quotes.get(b, {}).get("marketCap")
                if (
                    np.isfinite(corr)
                    and corr >= DEDUP_CORRELATION
                    and cap_a is not None
                    and cap_b is not None
                    and np.isfinite(float(cap_a))
                    and np.isfinite(float(cap_b))
                    and float(cap_a) > 0
                    and float(cap_b) > 0
                ):
                    removed.add(b if float(cap_a) >= float(cap_b) else a)
    return removed


def _screen_cache_path() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{price_cache.CACHE_DIR}/selection_screen_{today}.json"


def _universe_key(universe: dict[str, dict]) -> str:
    payload = json.dumps(universe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _read_cached_screen(expected_key: str | None = None) -> dict | None:
    path = _screen_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or (expected_key and data.get("universe_key") != expected_key):
            return None
        return data
    except (OSError, ValueError):
        return None


def run_screen(cfg: Config, force_refresh: bool = False) -> dict:
    """Run the quant selection screen for the S&P 100, cached daily.

    Returns:
        {as_of, universe_size, screened, skipped, sectors[], names[], from_cache}
        with names ranked by finance score (deduped twins flagged).
    """
    universe = load_universe()
    universe_key = _universe_key(universe)
    progress_log("screen", f"universe ready: {len(universe)} tickers")
    if not force_refresh:
        cached = _read_cached_screen(universe_key)
        if cached is not None:
            progress_log("screen", "screen cache hit — reusing today's quant results")
            return {**cached, "from_cache": True}

    from core.data.quotes import fetch_batch_quotes, fetch_fundamentals

    progress_log("screen", f"phase: price and volume history ({len(universe)} tickers)")
    prices, failures = load_universe_prices(universe)
    progress_log("screen", f"price history ready: {len(universe) - len(failures)} tickers")
    progress_log("screen", "phase: Yahoo quotes and fundamentals")
    quotes = fetch_batch_quotes(list(universe))
    fundamentals = fetch_fundamentals(list(universe))
    progress_log("screen", "phase: scoring names and sectors")
    for sym, row in fundamentals.items():
        merged = {k: v for k, v in quotes.get(sym, {}).items() if v is not None}
        merged.update(row)
        quotes[sym] = merged

    metrics: dict[str, dict] = {}
    for ticker in universe:
        if ticker not in prices:
            continue
        m = compute_name_metrics(prices[ticker], prices.get(f"{ticker}_volume"))
        if m is not None:
            metrics[ticker] = m

    scored = score_universe(metrics, quotes)
    sectors = sector_scores(scored, universe, metrics)
    removed = _dedup_near_duplicates(prices, universe, quotes)

    names = []
    for ticker in sorted(metrics, key=lambda s: scored[s]["finance"], reverse=True):
        e = universe[ticker]
        q = quotes.get(ticker, {})
        names.append({
            "ticker": ticker,
            "name": e["name"],
            "sector": e["sector"],
            "sub_industry": e["sub_industry"],
            "market_cap": q.get("marketCap"),
            "trailing_pe": q.get("trailingPE"),
            "revenue_growth": q.get("trailingAnnualRevenueGrowth"),
            "margins": q.get("profitMargins"),
            "momentum": round(scored[ticker]["momentum"], 4),
            "quality": round(scored[ticker]["quality"], 4),
            "finance_score": round(scored[ticker]["finance"], 4),
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics[ticker].items()},
            "deduped": ticker in removed,
        })

    screen = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "universe_key": universe_key,
        "universe_size": len(universe),
        "screened": len(metrics),
        "skipped": sorted(set(failures) | (set(universe) - set(metrics))),
        "sectors": sectors,
        "names": names,
    }
    os.makedirs(price_cache.CACHE_DIR, exist_ok=True)
    with open(_screen_cache_path(), "w") as f:
        json.dump(screen, f)
    return {**screen, "from_cache": False}
