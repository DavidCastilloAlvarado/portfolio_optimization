"""Weight bound resolution: per-ticker limits with global fallback."""

from config.defaults import Config


def build_asset_bounds(shares: list[str], cfg: Config) -> list:
    """Resolve (min, max) bounds for each ticker in order.

    Per-ticker entries win; a missing ticker or missing side falls back to the global limits.
    """
    gmin, gmax = cfg.w_limits
    per = {t.upper(): b for t, b in cfg.w_limits_per_ticker.items()}
    bounds = []
    for t in shares:
        p = per.get(t.upper())
        if p is None:
            bounds.append((gmin, gmax))
        else:
            lo, hi = p
            bounds.append((lo if lo is not None else gmin, hi if hi is not None else gmax))
    return bounds
