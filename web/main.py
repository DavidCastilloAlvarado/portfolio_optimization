"""FastAPI web UI for portfolio optimization."""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np

from config.defaults import Config
from data_loader import bulk_stocks, prepare_returns
from optimizer import optimize
from simulation import run_backtest
from web.renderer import build_optimization_result, build_backtest_result

app = FastAPI(title="Portfolio Optimizer", version="1.0.0")

# ── Static files ─────────────────────────────────────────────────────
_HTML = open("web/static/index.html").read()
_CSS = open("web/static/styles.css").read()
_JS = open("web/static/app.js").read()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    body = _HTML.replace("__CSS__", f"<style>\n{_CSS}\n</style>")
    body = body.replace("__JS__", _JS)
    return HTMLResponse(body)


@app.post("/optimize")
async def optimize_endpoint(
    resample: str = Form("none"),
    days: str = Form("720"),
    shares: str = Form(""),
    w_limits: str = Form("0.02,0.12"),
    min_variance: str = Form("off"),
    monto_usd: str = Form("10000"),
    monthly_delta: str = Form("300"),
    sim_days: str = Form("252"),
    risk_free_annual_perc: str = Form("5"),
):
    """Run the optimization pipeline with form parameters."""
    data = {
        "resample": resample,
        "days": days,
        "shares": shares,
        "w_limits": w_limits,
        "min_variance": min_variance,
        "monto_usd": monto_usd,
        "monthly_delta": monthly_delta,
        "sim_days": sim_days,
        "risk_free_annual_perc": risk_free_annual_perc,
    }

    cfg = Config.from_dict(data)

    # Data loading
    resample_val = cfg.get_resample()
    raw_data = bulk_stocks(cfg.shares, cfg.days)
    data_df, returns = prepare_returns(raw_data, resample=resample_val)

    names = data_df.columns.tolist()
    mean_returns = np.array(returns.mean())
    cov_returns = np.array(returns.cov())

    # Optimization
    weights, port_mean_val, port_std, strategy = optimize(
        mean_returns.copy(), cov_returns.copy(), cfg.risk_free, cfg.w_limits, cfg.min_variance,
    )

    # Build result
    result = build_optimization_result(
        weights, port_mean_val, port_std, strategy,
        names, cfg.monto_usd, cfg.risk_free,
    )

    # Backtest
    backtest = build_backtest_result(
        run_backtest(
            cfg.monto_usd, cfg.shares, weights, cfg.sim_days,
            cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
        )
    )
    if backtest:
        result["backtest"] = backtest

    return result
