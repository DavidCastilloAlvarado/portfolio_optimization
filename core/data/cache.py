"""Daily CSV price cache in temp/ ({TICKER}_{YYYY-MM-DD}.csv)."""

import glob
import os
from datetime import datetime, timezone

import pandas as pd

CACHE_DIR = "temp"


def cache_path(ticker: str) -> str:
    """Path of today's cache file for a ticker."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{CACHE_DIR}/{ticker}_{today_str}.csv"


def read_cache(ticker: str) -> pd.DataFrame | None:
    """Return today's cached prices, or None if not cached."""
    path = cache_path(ticker)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def write_cache(ticker: str, table: pd.DataFrame) -> None:
    """Persist prices to today's cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    table.to_csv(cache_path(ticker), index=False)


def latest_cache_file(ticker: str) -> str | None:
    """Most recent cached file for a ticker (any date), or None."""
    files = sorted(glob.glob(f"{CACHE_DIR}/{ticker}_*.csv"))
    return files[-1] if files else None


def read_prices(ticker: str) -> pd.Series | None:
    """Read a single ticker's price column from the most recent cache file."""
    path = latest_cache_file(ticker)
    if path is None:
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df[ticker]
