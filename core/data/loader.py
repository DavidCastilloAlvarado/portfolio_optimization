"""Market data access: Yahoo Finance fetch, JustETF fallback, daily CSV caching."""

import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import tqdm

from core.data.cache import read_cache, write_cache

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def is_isin(name: str) -> bool:
    """True if the symbol matches the ISIN pattern (2-letter country code + 9 alphanumerics + check digit)."""
    return bool(ISIN_RE.match(name.strip().upper()))


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
    dates = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps if ts is not None]

    return pd.DataFrame({
        "Date": dates,
        name.split(".")[0]: closes,
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


def load_table(name: str, init_time: int, end_time: int) -> pd.DataFrame:
    """Load daily close prices for a single symbol, with CSV caching in temp/.

    Yahoo Finance is the primary source; if it fails and the symbol is an ISIN,
    fall back to JustETF. Dates are normalized to midnight UTC so tables from
    different sources can be merged.
    """
    table = read_cache(name)
    if table is None:
        try:
            table = fetch_prices(name, init_time, end_time)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            if not is_isin(name):
                raise
            print(f"Yahoo failed for {name} ({exc}); falling back to JustETF")
            table = fetch_prices_justetf(name, init_time, end_time)
        write_cache(name, table)
    table["Date"] = pd.to_datetime(table["Date"], utc=True).dt.normalize()
    return table


def bulk_stocks(shares: list, days_back: int) -> pd.DataFrame:
    """Load and merge daily prices for all tickers into a single DataFrame."""
    init_time, end_time = get_unix_time(days_back)
    data = None
    for i, share in tqdm.tqdm(enumerate(shares), total=len(shares)):
        print(share)
        if i == 0:
            data = load_table(share, init_time, end_time)
        else:
            temp = load_table(share, init_time, end_time)
            data = data.merge(temp, on=["Date"])
    return data


def prepare_returns(data: pd.DataFrame, resample: str | None = None) -> tuple:
    """Sort data by date, interpolate gaps, and compute daily returns.

    Args:
        data:      DataFrame with a 'Date' column and one column per ticker.
        resample:  None for daily, 'week' or 'month' for aggregated periods.

    Returns:
        (data, returns) — the cleaned price DataFrame and the returns DataFrame.
    """
    if resample == "week":
        data["step"] = data.Date.apply(
            lambda x: f"{x.isocalendar()[1]}-{x.isocalendar()[0]}-{x.month}"
        )
        data = data.groupby("step").last()
    elif resample == "month":
        data["step"] = data.Date.apply(
            lambda x: f"{x.isocalendar()[0]}-{x.month}"
        )
        data = data.groupby("step").last()

    data = data.sort_values("Date", ascending=False).set_index("Date")
    data.interpolate(method="time", limit_direction="backward", inplace=True)
    returns = data.pct_change(periods=-1)
    return data, returns
