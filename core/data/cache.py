"""Daily CSV price cache in temp/ ({TICKER}_{YYYY-MM-DD}.csv)."""

import glob
import os
import tempfile
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
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Date" not in df.columns or ticker not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        return df.dropna(subset=["Date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def write_cache(ticker: str, table: pd.DataFrame) -> None:
    """Persist prices to today's cache file atomically."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    destination = cache_path(ticker)
    fd, temporary = tempfile.mkstemp(prefix=".cache-", suffix=".csv", dir=CACHE_DIR)
    os.close(fd)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def latest_cache_file(ticker: str) -> str | None:
    """Most recent cached file for a ticker (any date), or None."""
    files = sorted(glob.glob(f"{CACHE_DIR}/{ticker}_*.csv"))
    return files[-1] if files else None


def read_prices(ticker: str) -> pd.Series | None:
    """Read and merge all valid cached price history for one ticker."""
    files = sorted(glob.glob(f"{CACHE_DIR}/{ticker}_*.csv"))
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if "Date" not in df.columns or ticker not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.normalize()
            frames.append(df.dropna(subset=["Date"])[["Date", ticker]])
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return merged.set_index("Date")[ticker]
