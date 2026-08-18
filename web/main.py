"""FastAPI web UI for portfolio optimization."""

import traceback
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

from config.defaults import Config
from core.pipeline import run_pipeline
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
    w_limits_per_ticker: str = Form(""),
    min_variance: str = Form("off"),
    monto_usd: str = Form("10000"),
    monthly_delta: str = Form("300"),
    sim_days: str = Form("252"),
    risk_free_annual_perc: str = Form("5"),
):
    """Run the optimization pipeline with form parameters."""
    try:
        data = {
            "resample": resample,
            "days": days,
            "shares": shares,
            "w_limits": w_limits,
            "w_limits_per_ticker": w_limits_per_ticker,
            "min_variance": min_variance,
            "monto_usd": monto_usd,
            "monthly_delta": monthly_delta,
            "sim_days": sim_days,
            "risk_free_annual_perc": risk_free_annual_perc,
        }

        cfg = Config.from_dict(data)
        result = run_pipeline(cfg)

        optimization = build_optimization_result(
            result["weights"], result["mean"], result["std"], result["strategy"],
            result["names"], cfg.monto_usd, cfg.risk_free,
        )

        backtest = build_backtest_result(result.get("backtest"))
        if backtest:
            optimization["backtest"] = backtest

        return optimization

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "traceback": tb},
        )
