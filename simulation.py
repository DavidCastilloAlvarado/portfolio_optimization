import random
import glob
import os

import numpy as np
import pandas as pd
from datetime import datetime, timezone


def final_wallet(monto, periodos, mu,sigma):
    for _ in range(periodos):
        val = random.normalvariate(mu, sigma)
        monto += monto*val
    return monto


def print_rendimiento(MONTO, Ts, MU, STD, TOTAL_SIM):
    """# MU = 0.001409
    # STD = 0.01276
    # Ts = 360
    # MONTO = 5000
    # TOTAL_SIM = 1000"""
    distr = [(final_wallet(MONTO, Ts, MU ,STD  )-MONTO)/MONTO*100 for _ in range(TOTAL_SIM)]
    distr = pd.Series(distr)
    print('======== RENDIMIENTO (%) ========')
    print(distr.describe())


def _load_backtest_prices(shares, n_days):
    """Load price data from cached CSVs and return the last n_days rows."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prices = pd.DataFrame()

    for share in shares:
        cache_file = f"temp/{share}_{today_str}.csv"
        if not os.path.exists(cache_file):
            files = sorted(glob.glob(f"temp/{share}_*.csv"))
            if files:
                cache_file = files[-1]
            else:
                print(f"  ⚠ No CSV found for {share}, skipping.")
                continue
        df = pd.read_csv(cache_file, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        prices[share] = df[share]

    prices = prices.tail(n_days).dropna()
    return prices


def _compute_stats(values, total_invested, portfolio_returns, risk_free_annual):
    """Compute backtest statistics from the equity curve."""
    final_return_pct = (values[-1] - total_invested) / total_invested * 100
    cummax = np.maximum.accumulate(values)
    drawdown = (cummax - values) / cummax * 100
    max_dd = drawdown.max()
    ret_series = pd.Series(portfolio_returns)
    daily_rf = (1 + risk_free_annual) ** (1/252) - 1
    sharpe = (
        (ret_series.mean() - daily_rf) / ret_series.std() * np.sqrt(252)
        if ret_series.std() > 0 else 0.0
    )
    return final_return_pct, max_dd, sharpe, ret_series


def print_rendimiento_backtest(monto, shares, weights, n_days,
                               risk_free_annual=0.0, monthly_delta=0.0):
    """Backward simulation using real historical data from CSV files.

    Prints two modes:
      1. Lump-sum: single initial investment of `monto`.
      2. DCA: initial investment of `monto` + `monthly_delta` added every
         ~21 trading days, distributed with the same portfolio weights.

    Args:
        monto:            Initial investment amount (USD)
        shares:           List of ticker symbols
        weights:          Array/list of portfolio weights per share
        n_days:           Number of trading days to look back
        risk_free_annual: Annual risk-free rate as decimal (e.g. 0.05 for 5%)
        monthly_delta:    USD amount added every month (0 = lump-sum only)
    """
    prices = _load_backtest_prices(shares, n_days)
    if prices.empty or len(prices) < 2:
        print("  ⚠ Not enough price data for backtest.")
        return

    daily_returns = prices.pct_change().dropna()
    w = np.array(weights, dtype=float)
    portfolio_returns = daily_returns.values @ w
    trading_days = len(portfolio_returns)

    # ── Mode 1: Lump-sum ──
    portfolio_value = monto
    values = [portfolio_value]
    for r in portfolio_returns:
        portfolio_value *= (1 + r)
        values.append(portfolio_value)
    values_ls = np.array(values)

    ret_pct_ls, max_dd_ls, sharpe_ls, ret_s = _compute_stats(
        values_ls, monto, portfolio_returns, risk_free_annual)

    print(f"======== BACKTEST LUMP-SUM (last {trading_days} trading days) ========")
    print(f"  Initial investment : {monto:.2f} USD")
    print(f"  Final value        : {values_ls[-1]:.2f} USD")
    print(f"  Total return       : {ret_pct_ls:+.2f}%")
    print(f"  Max drawdown       : {max_dd_ls:.2f}%")
    print(f"  Sharpe (annualized): {sharpe_ls:.4f}")

    # ── Mode 2: DCA (initial + monthly contributions) ──
    if monthly_delta > 0:
        TRADING_DAYS_PER_MONTH = 21
        portfolio_value = monto
        total_invested = monto
        values_dca = [portfolio_value]

        for i, r in enumerate(portfolio_returns):
            portfolio_value *= (1 + r)
            # Add monthly delta every ~21 trading days (starting from day 21)
            if (i + 1) % TRADING_DAYS_PER_MONTH == 0:
                portfolio_value += monthly_delta
                total_invested += monthly_delta
            values_dca.append(portfolio_value)

        values_dca = np.array(values_dca)
        ret_pct_dca, max_dd_dca, sharpe_dca, _ = _compute_stats(
            values_dca, total_invested, portfolio_returns, risk_free_annual)

        months_contributed = trading_days // TRADING_DAYS_PER_MONTH
        print(f"======== BACKTEST DCA +{monthly_delta:.0f} USD/month (last {trading_days} trading days) ========")
        print(f"  Initial investment : {monto:.2f} USD")
        print(f"  Monthly addition   : {monthly_delta:.2f} USD x {months_contributed} months")
        print(f"  Total invested     : {total_invested:.2f} USD")
        print(f"  Final value        : {values_dca[-1]:.2f} USD")
        print(f"  Total return       : {ret_pct_dca:+.2f}%")
        print(f"  Max drawdown       : {max_dd_dca:.2f}%")
        print(f"  Sharpe (annualized): {sharpe_dca:.4f}")