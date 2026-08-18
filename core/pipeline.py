"""Optimization pipeline: the single computation flow shared by CLI and web."""

import numpy as np

from config.defaults import Config
from core.data.loader import bulk_stocks, prepare_returns
from core.optimization.bounds import build_asset_bounds
from core.optimization.solvers import optimize
from core.simulation.backtest import run_backtest


def run_pipeline(cfg: Config) -> dict:
    """Run the full pipeline: data → optimization → backtest.

    Returns a dict with names, weights, mean, std, strategy and the backtest result.
    """
    resample_val = cfg.get_resample()
    raw_data = bulk_stocks(cfg.shares, cfg.days)
    data_df, returns = prepare_returns(raw_data, resample=resample_val)

    names = data_df.columns.tolist()
    mean_returns = np.array(returns.mean())
    cov_returns = np.array(returns.cov())

    weights, port_mean_val, port_std, strategy = optimize(
        mean_returns.copy(), cov_returns.copy(),
        cfg.risk_free, build_asset_bounds(names, cfg), cfg.min_variance,
    )

    result = {
        "names": names,
        "records": int(len(data_df)),
        "weights": weights.tolist(),
        "mean": float(port_mean_val),
        "std": float(port_std),
        "strategy": strategy,
    }

    backtest = run_backtest(
        cfg.monto_usd, cfg.shares, weights, cfg.sim_days,
        cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
    )
    if backtest:
        result["backtest"] = backtest

    return result
