import random

import numpy as np
import pandas as pd
from typing import Callable

from core.data.cache import read_prices


def final_wallet(monto, periodos, mu, sigma):
    for _ in range(periodos):
        val = random.normalvariate(mu, sigma)
        monto += monto * val
    return monto


def print_rendimiento(MONTO, Ts, MU, STD, TOTAL_SIM):
    distr = [(final_wallet(MONTO, Ts, MU, STD) - MONTO) / MONTO * 100 for _ in range(TOTAL_SIM)]
    distr = pd.Series(distr)
    print('======== RENDIMIENTO (%) ========')
    print(distr.describe())


def _load_backtest_prices(shares, n_days):
    """Load price data from cached CSVs and return the last n_days rows."""
    prices = pd.DataFrame()

    for share in shares:
        series = read_prices(share)
        if series is None:
            print(f"  ⚠ No CSV found for {share}, skipping.")
            continue
        prices[share] = series

    prices = prices.tail(n_days).dropna()
    return prices


def _compute_stats(values, total_invested, portfolio_returns, risk_free_annual):
    """Compute backtest statistics from the equity curve."""
    final_return_pct = (values[-1] - total_invested) / total_invested * 100
    cummax = np.maximum.accumulate(values)
    drawdown = (cummax - values) / cummax * 100
    max_dd = drawdown.max()
    ret_series = pd.Series(portfolio_returns)
    daily_rf = (1 + risk_free_annual) ** (1 / 252) - 1
    sharpe = (
        (ret_series.mean() - daily_rf) / ret_series.std() * np.sqrt(252)
        if ret_series.std() > 0 else 0.0
    )
    return final_return_pct, max_dd, sharpe, ret_series


def run_backtest(
    monto,
    shares,
    weights,
    n_days,
    risk_free_annual=0.0,
    monthly_delta=0.0,
    price_loader: Callable | None = None,
):
    """Run lump-sum and optional DCA backtests using cached historical prices."""
    if monto <= 0 or monthly_delta < 0 or n_days < 2:
        return {"error": "Investment and simulation inputs must be positive."}
    if len(shares) != len(weights) or not shares:
        return {"error": "Shares and weights must contain the same non-empty assets."}
    if not np.all(np.isfinite(np.asarray(weights, dtype=float))):
        return {"error": "Portfolio weights must be finite."}

    loader = price_loader or _load_backtest_prices
    prices = loader(shares, n_days)
    if not isinstance(prices, pd.DataFrame) or prices.empty or len(prices) < 2:
        return {"error": "Not enough price data for backtest."}

    missing = [share for share, weight in zip(shares, weights) if weight > 0 and share not in prices.columns]
    if missing:
        return {"error": f"Missing price data for invested assets: {', '.join(missing)}."}
    available = [share for share in shares if share in prices.columns]
    prices = prices[available].sort_index().dropna()
    if len(prices) < 2:
        return {"error": "Not enough aligned price data for backtest."}

    daily_returns = prices.pct_change().dropna()
    w_map = dict(zip(shares, weights))
    w = np.array([w_map.get(c, 0.0) for c in daily_returns.columns], dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return {"error": "No usable weights for the available price data."}
    w = w / w_sum
    portfolio_returns = daily_returns.values @ w
    trading_days = len(portfolio_returns)

    # ── Mode 1: Lump-sum ──
    portfolio_value = monto
    values = [portfolio_value]
    for r in portfolio_returns:
        portfolio_value *= (1 + r)
        values.append(portfolio_value)
    values_ls = np.array(values)

    ret_pct_ls, max_dd_ls, sharpe_ls, _ = _compute_stats(
        values_ls, monto, portfolio_returns, risk_free_annual)

    result = {
        "lump_sum": {
            "trading_days": trading_days,
            "initial_investment": monto,
            "final_value": float(values_ls[-1]),
            "total_return_pct": float(ret_pct_ls),
            "max_drawdown_pct": float(max_dd_ls),
            "sharpe_annualized": float(sharpe_ls),
        }
    }

    # ── Mode 2: DCA ──
    if monthly_delta > 0:
        TRADING_DAYS_PER_MONTH = 21
        portfolio_value = monto
        total_invested = monto
        values_dca = [portfolio_value]
        dca_returns = []

        for i, r in enumerate(portfolio_returns):
            previous_value = portfolio_value
            portfolio_value *= (1 + r)
            contribution = monthly_delta if (i + 1) % TRADING_DAYS_PER_MONTH == 0 else 0.0
            if contribution:
                portfolio_value += contribution
                total_invested += contribution
            dca_returns.append(
                (portfolio_value - contribution) / previous_value - 1.0
                if previous_value > 0 else 0.0
            )
            values_dca.append(portfolio_value)

        values_dca = np.array(values_dca)
        ret_pct_dca, max_dd_dca, sharpe_dca, _ = _compute_stats(
            values_dca, total_invested, dca_returns, risk_free_annual)

        months_contributed = trading_days // TRADING_DAYS_PER_MONTH
        result["dca"] = {
            "trading_days": trading_days,
            "initial_investment": monto,
            "monthly_addition": monthly_delta,
            "months_contributed": months_contributed,
            "total_invested": total_invested,
            "final_value": float(values_dca[-1]),
            "total_return_pct": float(ret_pct_dca),
            "max_drawdown_pct": float(max_dd_dca),
            "sharpe_annualized": float(sharpe_dca),
        }

    return result
