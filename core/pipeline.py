"""Optimization pipeline: the single computation flow shared by CLI and web."""

import numpy as np

from config.defaults import Config
from core.data.loader import bulk_stocks, prepare_returns, ticker_for
from core.optimization.bounds import build_asset_bounds
from core.optimization.solvers import optimize
from core.simulation.backtest import run_backtest
from models.blm import run_blm


def _llm_views_report(cfg: Config, blm_result: dict) -> dict:
    assets = []
    for name in blm_result["view_names"]:
        v = blm_result["views"][name]
        assets.append({
            "ticker": name,
            "symbol": ticker_for(name),
            "predicted_daily_return_pct": float(v["q"] * 100),
            "view_std_pct": float(np.sqrt(v["omega"]) * 100),
            "samples_ok": int(v["n_ok"]),
            "samples_pct": [float(s * 100) for s in v["samples"]],
        })
    return {
        "model": blm_result["model"],
        "from_cache": bool(blm_result["from_cache"]),
        "degraded": bool(blm_result["degraded"]),
        "tau": float(cfg.llm_tau),
        "samples_per_asset": int(cfg.llm_samples),
        "horizon_days": int(cfg.blm_horizon_days),
        "assets": assets,
    }


def run_pipeline(cfg: Config) -> dict:
    """Run the full pipeline: data → optimization → backtest.

    Returns a dict with names, weights, mean, std, strategy and the backtest result.
    """
    resample_val = cfg.get_resample()
    raw_data = bulk_stocks(cfg.shares, cfg.days)
    data_df, returns = prepare_returns(raw_data, resample=resample_val)

    names = data_df.columns.tolist()
    strategy_resolved = cfg.resolve_strategy()
    llm_report = None

    if strategy_resolved == "llm_bl":
        _, daily_returns = prepare_returns(raw_data, resample=None)
        daily_cov = np.array(daily_returns.cov())
        blm = run_blm(daily_returns, names, daily_cov, cfg)
        weights, port_mean_val, port_std, _ = optimize(
            blm["posterior_mean"].copy(), daily_cov.copy(),
            cfg.risk_free, build_asset_bounds(names, cfg),
        )
        if blm["degraded"]:
            strategy = "Black-Litterman Optimization (equilibrium fallback)"
        else:
            strategy = "Black-Litterman Optimization (LLM views)"
        llm_report = _llm_views_report(cfg, blm)
    else:
        mean_returns = np.array(returns.mean())
        cov_returns = np.array(returns.cov())
        weights, port_mean_val, port_std, strategy = optimize(
            mean_returns.copy(), cov_returns.copy(),
            cfg.risk_free, build_asset_bounds(names, cfg),
            strategy=strategy_resolved, kelly_fraction=cfg.kelly_fraction,
        )

    result = {
        "names": names,
        "records": int(len(data_df)),
        "weights": weights.tolist(),
        "mean": float(port_mean_val),
        "std": float(port_std),
        "strategy": strategy,
    }
    if llm_report is not None:
        result["llm_views"] = llm_report

    backtest = run_backtest(
        cfg.monto_usd, cfg.shares, weights, cfg.sim_days,
        cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
    )
    if backtest:
        result["backtest"] = backtest

    return result
