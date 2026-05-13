"""Server-side result building for the portfolio optimizer."""

import numpy as np
from typing import Any


def build_optimization_result(
    weights: list[float],
    port_mean_val: float,
    port_std: float,
    strategy: str,
    names: list[str],
    monto_usd: float,
    risk_free: float,
) -> dict[str, Any]:
    """Build the optimization result dict from optimizer output."""
    weights_resp = []
    for name, w in zip(names, weights):
        weights_resp.append({
            "ticker": name,
            "weight_pct": float(w * 100),
            "usd": float(w * monto_usd),
        })

    port_return_daily = float(port_mean_val)
    port_std_daily = float(port_std)
    if port_std_daily > 0:
        sharpe_daily = (port_return_daily - risk_free) / port_std_daily
        sharpe_annual = sharpe_daily * np.sqrt(365)
    else:
        sharpe_annual = 0.0

    return {
        "strategy": strategy,
        "weights": weights_resp,
        "portfolio_return_pct": float(port_return_daily * 365 * 100),
        "portfolio_std_pct": float(port_std_daily * np.sqrt(365) * 100),
        "sharpe_ratio": float(sharpe_annual),
    }


def build_backtest_result(backtest: dict[str, Any] | None) -> dict[str, Any] | None:
    """Filter backtest results, returning None on error."""
    if backtest and "error" not in backtest:
        return backtest
    return None
