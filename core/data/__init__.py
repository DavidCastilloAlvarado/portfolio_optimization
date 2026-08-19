"""Market data access: Yahoo Finance fetch, JustETF fallback, CSV cache, return preparation."""

from core.data.cache import cache_path, latest_cache_file, read_cache, read_prices, write_cache
from core.data.loader import (
    bulk_stocks,
    fetch_prices,
    fetch_prices_justetf,
    get_unix_time,
    is_isin,
    load_table,
    prepare_returns,
)

__all__ = [
    "cache_path", "latest_cache_file", "read_cache", "read_prices", "write_cache",
    "bulk_stocks", "fetch_prices", "fetch_prices_justetf", "get_unix_time",
    "is_isin", "load_table", "prepare_returns",
]
