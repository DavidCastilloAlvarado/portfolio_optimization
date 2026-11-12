"""Optimization pipeline: the single computation flow shared by CLI and web."""

import time

import numpy as np

from config.defaults import Config
from core.data.loader import bulk_stocks, prepare_returns, ticker_for
from core.optimization.bounds import build_asset_bounds
from core.optimization.solvers import optimize
from core.progress import elapsed, progress_log
from core.simulation.backtest import run_backtest
from models.blm import run_blm


def _llm_views_report(cfg: Config, blm_result: dict) -> dict:
    assets = []
    for name in blm_result["view_names"]:
        v = blm_result["views"][name]
        entry = {
            "ticker": name,
            "symbol": ticker_for(name),
            "predicted_daily_return_pct": float(v["q"] * 100),
            "view_std_pct": float(np.sqrt(v["omega"]) * 100),
            "samples_ok": int(v["n_ok"]),
            "samples_pct": [float(s * 100) for s in v["samples"]],
        }
        details = v.get("details") or []
        if details:
            sources = []
            for d in details:
                for url in d.get("sources", []):
                    if url not in sources:
                        sources.append(url)
            entry.update({
                "rationale": details[len(details) // 2].get("rationale", ""),
                "sources": sources[:12],
                "confidence": float(np.mean([d.get("confidence", 0.5) for d in details])),
            })
        assets.append(entry)
    agent_sourced = any(v.get("details") for v in blm_result["views"].values())
    return {
        "model": blm_result["model"],
        "view_engine": "agent" if agent_sourced else "direct",
        "from_cache": bool(blm_result["from_cache"]),
        "degraded": bool(blm_result["degraded"]),
        "tau": float(cfg.llm_tau),
        "samples_per_asset": int(cfg.llm_samples),
        "horizon_days": int(cfg.blm_horizon_days),
        "assets": assets,
    }


def run_pipeline(cfg: Config) -> dict:
    """Run the full pipeline: data → optimization → backtest."""
    if not cfg.shares:
        raise ValueError("at least one ticker is required")
    resample_val = cfg.get_resample()
    t_data = time.monotonic()
    progress_log("pipeline", f"phase: data — loading {len(cfg.shares)} tickers x {cfg.days}d")
    raw_data = bulk_stocks(cfg.shares, cfg.days)
    data_df, returns = prepare_returns(raw_data, resample=resample_val)

    names = data_df.columns.tolist()
    strategy_resolved = cfg.resolve_strategy()
    progress_log("pipeline", f"data loaded: {len(data_df)} rows x {len(names)} tickers (+{elapsed(t_data)})")
    llm_report = None

    if strategy_resolved == "llm_bl":
        progress_log("pipeline", "phase: LLM-Black-Litterman (agent views)")
        _, daily_returns = prepare_returns(raw_data, resample=None)
        daily_cov = np.array(daily_returns.cov())
        blm = run_blm(daily_returns, names, daily_cov, cfg)
        t_opt = time.monotonic()
        progress_log("pipeline", "phase: optimization (Sharpe MVO on BL posterior)")
        weights, port_mean_val, port_std, _ = optimize(
            blm["posterior_mean"].copy(), daily_cov.copy(),
            cfg.risk_free, build_asset_bounds(names, cfg),
        )
        progress_log("pipeline", f"optimization done (+{elapsed(t_opt)})")
        if blm["degraded"]:
            strategy = "Black-Litterman Optimization (equilibrium fallback)"
        else:
            strategy = "Black-Litterman Optimization (LLM views)"
        llm_report = _llm_views_report(cfg, blm)
    else:
        mean_returns = np.array(returns.mean())
        cov_returns = np.array(returns.cov())
        t_opt = time.monotonic()
        progress_log("pipeline", f"phase: optimization (strategy {strategy_resolved})")
        weights, port_mean_val, port_std, strategy = optimize(
            mean_returns.copy(), cov_returns.copy(),
            cfg.risk_free, build_asset_bounds(names, cfg),
            strategy=strategy_resolved, kelly_fraction=cfg.kelly_fraction,
        )
        progress_log("pipeline", f"optimization done (+{elapsed(t_opt)})")

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

    t_bt = time.monotonic()
    progress_log("pipeline", "phase: backtest")
    backtest = run_backtest(
        cfg.monto_usd, names, weights, cfg.sim_days,
        cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
    )
    if backtest:
        result["backtest"] = backtest
        progress_log("pipeline", f"backtest done (+{elapsed(t_bt)})")

    return result
