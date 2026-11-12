"""Market data access: Yahoo Finance fetch, JustETF fallback, daily CSV caching."""

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import tqdm

from core.data.cache import read_cache, write_cache
from core.progress import progress_log

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

ISIN_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "isin_ticker_map.json"


def _load_isin_ticker_map() -> dict[str, str]:
    try:
        raw = ISIN_MAP_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"ISIN map file not found: {ISIN_MAP_PATH}. Create it with one \"ISIN\": \"TICKER\" pair per line."
        )
    try:
        raw_map = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in ISIN map {ISIN_MAP_PATH}: {exc}") from exc
    if not isinstance(raw_map, dict):
        raise ValueError(f"ISIN map {ISIN_MAP_PATH} must be a JSON object with \"ISIN\": \"TICKER\" pairs")
    return {str(k).strip().upper(): str(v).strip().upper() for k, v in raw_map.items()}


ISIN_TICKER_MAP: dict[str, str] = _load_isin_ticker_map()


def is_isin(name: str) -> bool:
    """True if the symbol matches the ISIN pattern (2-letter country code + 9 alphanumerics + check digit)."""
    return bool(ISIN_RE.match(name.strip().upper()))


def ticker_for(name: str) -> str | None:
    """Display ticker mapped to an ISIN for easy reading, or None for plain tickers."""
    return ISIN_TICKER_MAP.get(name.strip().upper())


def get_unix_time(days_back: int) -> tuple:
    """Return (init_time, end_time) as unix timestamps, looking back `days_back` calendar days."""
    init_time = datetime.now() - timedelta(days=days_back)
    end_time = datetime.now()

    def unix(dt: datetime) -> int:
        return int(dt.replace(tzinfo=timezone.utc).timestamp())

    return unix(init_time), unix(end_time)


def fetch_prices(name: str, init_time: int, end_time: int) -> pd.DataFrame:
    """Fetch daily close prices for a ticker from Yahoo Finance Query v8."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{name}"
        f"?period1={init_time}&period2={end_time}&interval=1d"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    result = data["chart"]["result"]
    if not result or result[0] is None:
        error_msg = data["chart"].get("error", "Unknown error from Yahoo Finance")
        raise ValueError(f"Failed to fetch data for ticker '{name}': {error_msg}")
    result = result[0]
    timestamps = result.get("timestamp", [])
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    if not timestamps:
        raise ValueError(f"No price data returned for ticker '{name}' — ticker may be invalid or delisted.")
    pairs = []
    for timestamp, close in zip(timestamps, closes):
        try:
            value = float(close)
        except (TypeError, ValueError):
            continue
        if timestamp is None or not math.isfinite(value):
            continue
        pairs.append((datetime.fromtimestamp(timestamp, tz=timezone.utc), value))
    if not pairs:
        raise ValueError(f"No usable price data returned for ticker '{name}'.")

    dates, values = zip(*pairs)
    return pd.DataFrame({
        "Date": list(dates),
        name.split(".")[0]: list(values),
    })


def fetch_prices_justetf(name: str, init_time: int, end_time: int) -> pd.DataFrame:
    """Fetch daily close prices for an ISIN from the JustETF performance-chart API."""
    date_from = datetime.fromtimestamp(init_time, tz=timezone.utc).strftime("%Y-%m-%d")
    date_to = datetime.fromtimestamp(end_time, tz=timezone.utc).strftime("%Y-%m-%d")
    url = (
        f"https://www.justetf.com/api/etfs/{name}/performance-chart"
        f"?locale=es&currency=USD&valuesType=MARKET_VALUE&reduceData=false"
        f"&includeDividends=false&features=DIVIDENDS&dateFrom={date_from}&dateTo={date_to}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    series = data.get("series", [])
    if not series:
        raise ValueError(f"No price data returned for ISIN '{name}' — ISIN may be invalid or the fund may be delisted.")
    dates = [pd.to_datetime(item["date"], utc=True) for item in series]
    values = [item["value"]["raw"] for item in series]
    return pd.DataFrame({"Date": dates, name: values})


def _requested_window(init_time: int, end_time: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp.fromtimestamp(init_time, tz="UTC").normalize(),
        pd.Timestamp.fromtimestamp(end_time, tz="UTC").normalize(),
    )


def _cache_covers_window(table: pd.DataFrame, init_time: int, end_time: int) -> bool:
    if table.empty or "Date" not in table:
        return False
    start, end = _requested_window(init_time, end_time)
    span_days = (end - start).days
    dates = pd.to_datetime(table["Date"], utc=True).dropna().dt.normalize()
    if dates.empty:
        return False
    if span_days <= 1:
        return True
    tolerance = pd.Timedelta(days=4)
    return dates.min() <= start and dates.max() >= end - tolerance


def load_table(name: str, init_time: int, end_time: int) -> pd.DataFrame:
    """Load daily close prices for a single symbol, with CSV caching in temp/."""
    table = read_cache(name)
    cached_table = table
    if table is not None:
        table["Date"] = pd.to_datetime(table["Date"], utc=True).dt.normalize()
        if _cache_covers_window(table, init_time, end_time):
            start, end = _requested_window(init_time, end_time)
            if (end - start).days > 1:
                table = table[table["Date"].between(start, end)].copy()
            print(f"{name}: cache hit ({len(table)} rows)", flush=True)
        else:
            table = None
            print(f"{name}: cache does not cover requested window; refreshing", flush=True)
    if table is None:
        try:
            table = fetch_prices(name, init_time, end_time)
            print(f"{name}: yahoo fetched ({len(table)} rows)", flush=True)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            if cached_table is not None and not is_isin(name):
                table = cached_table
                print(f"{name}: using stale cache after refresh failure ({exc})", flush=True)
            elif not is_isin(name):
                raise
            else:
                print(f"Yahoo failed for {name} ({exc}); falling back to JustETF", flush=True)
                table = fetch_prices_justetf(name, init_time, end_time)
                print(f"{name}: justetf fetched ({len(table)} rows)", flush=True)
        write_cache(name, table)
    table["Date"] = pd.to_datetime(table["Date"], utc=True).dt.normalize()
    return table


def bulk_stocks(shares: list, days_back: int) -> pd.DataFrame:
    """Load and merge daily prices for all tickers into a single DataFrame."""
    init_time, end_time = get_unix_time(days_back)
    data = None
    for i, share in tqdm.tqdm(enumerate(shares), total=len(shares)):
        progress_log("data", f"loading {share} ({i + 1}/{len(shares)})")
        if i == 0:
            data = load_table(share, init_time, end_time)
        else:
            temp = load_table(share, init_time, end_time)
            data = data.merge(temp, on=["Date"])
    return data


def prepare_returns(data: pd.DataFrame, resample: str | None = None) -> tuple:
    """Clean prices, optionally resample them, and calculate returns."""
    if "Date" not in data.columns:
        raise ValueError("price data must contain a Date column")
    if resample not in (None, "week", "month"):
        raise ValueError("resample must be None, 'week', or 'month'")

    cleaned = data.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], utc=True, errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"]).sort_values("Date")
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="last")
    if cleaned.empty:
        raise ValueError("price data contains no valid dates")

    price_columns = [column for column in cleaned.columns if column != "Date"]
    if not price_columns:
        raise ValueError("price data contains no asset columns")
    indexed = cleaned.set_index("Date")[price_columns]
    if resample == "week":
        indexed = indexed.resample("W").last()
    elif resample == "month":
        indexed = indexed.resample("ME").last()
    indexed = indexed.dropna(how="all")
    indexed = indexed.interpolate(method="time", limit_direction="both")
    if indexed.empty:
        raise ValueError("price data contains no usable observations")

    prices = indexed.sort_index(ascending=False)
    returns = prices.pct_change(periods=-1)
    return prices, returns
