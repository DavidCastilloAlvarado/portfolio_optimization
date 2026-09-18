"""FastAPI web UI for portfolio optimization."""

import json
import traceback
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

from config.defaults import Config
from core.data.loader import ISIN_TICKER_MAP
from core.pipeline import run_pipeline
from web.renderer import build_optimization_result, build_backtest_result

app = FastAPI(title="Portfolio Optimizer", version="1.0.0")

# ── Static files ─────────────────────────────────────────────────────
_HTML = open("web/static/index.html").read()
_CSS = open("web/static/styles.css").read()
_JS = open("web/static/app.js").read()
_ISIN_MAP = json.dumps(ISIN_TICKER_MAP)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    body = _HTML.replace("__CSS__", f"<style>\n{_CSS}\n</style>")
    body = body.replace("__ISIN_MAP__", _ISIN_MAP)
    body = body.replace("__JS__", _JS)
    return HTMLResponse(body)


@app.post("/optimize")
def optimize_endpoint(
    resample: str = Form("none"),
    days: str = Form("720"),
    shares: str = Form(""),
    w_limits: str = Form("0.02,0.12"),
    w_limits_per_ticker: str = Form(""),
    min_variance: str = Form("off"),
    strategy: str = Form("sharpe"),
    kelly_fraction: str = Form("0.5"),
    llm_samples: str = Form("5"),
    llm_tau: str = Form("0.0385"),
    blm_prior: str = Form("equal"),
    blm_horizon_days: str = Form("14"),
    blm_history_days: str = Form("56"),
    llm_view_engine: str = Form("agent"),
    llm_reasoning_effort: str = Form("medium"),
    llm_agent_websearch: str = Form("on"),
    llm_agent_max_tokens: str = Form("32768"),
    llm_agent_recursion_limit: str = Form("40"),
    monto_usd: str = Form("10000"),
    monthly_delta: str = Form("300"),
    sim_days: str = Form("252"),
    risk_free_annual_perc: str = Form("5"),
):
    """Run the optimization pipeline with form parameters.

    Sync endpoint on purpose: FastAPI runs it in the threadpool, which allows the
    LLM research agent to open its own asyncio event loop for the MCP session.
    """
    try:
        data = {
            "resample": resample,
            "days": days,
            "shares": shares,
            "w_limits": w_limits,
            "w_limits_per_ticker": w_limits_per_ticker,
            "min_variance": min_variance,
            "strategy": strategy,
            "kelly_fraction": kelly_fraction,
            "llm_samples": llm_samples,
            "llm_tau": llm_tau,
            "blm_prior": blm_prior,
            "blm_horizon_days": blm_horizon_days,
            "blm_history_days": blm_history_days,
            "llm_view_engine": llm_view_engine,
            "llm_reasoning_effort": llm_reasoning_effort,
            "llm_agent_websearch": llm_agent_websearch,
            "llm_agent_max_tokens": llm_agent_max_tokens,
            "llm_agent_recursion_limit": llm_agent_recursion_limit,
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
        if result.get("llm_views"):
            optimization["llm_views"] = result["llm_views"]

        return optimization

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "traceback": tb},
        )
