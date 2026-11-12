"""FastAPI web UI for portfolio optimization."""

import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

from config.defaults import Config
from core.data.loader import ISIN_TICKER_MAP
from core.pipeline import run_pipeline
from web.progress_jobs import progress_jobs
from web.renderer import build_optimization_result, build_backtest_result

app = FastAPI(title="Portfolio Optimizer", version="1.0.0")
logger = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _read_static(name: str) -> str:
    return (_STATIC_DIR / name).read_text(encoding="utf-8")


_HTML = _read_static("index.html")
_CSS = _read_static("styles.css")
_PROGRESS_JS = _read_static("progress.js")
_JS = _PROGRESS_JS + "\n" + _read_static("app.js")
_SELECT_HTML = _read_static("select.html")
_SELECT_JS = _PROGRESS_JS + "\n" + _read_static("select.js")
_ISIN_MAP = json.dumps(ISIN_TICKER_MAP)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    return response


def _error_response(exc: Exception, operation: str, status_code: int = 400) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]
    logger.exception("%s failed [request_id=%s]", operation, request_id)
    message = str(exc) if isinstance(exc, ValueError) else f"{operation} failed. Check the server logs."
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "request_id": request_id},
    )


def _busy_response() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "Another analysis is already running. Please retry shortly."},
        headers={"Retry-After": "5"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    body = _HTML.replace("__CSS__", f"<style>\n{_CSS}\n</style>")
    body = body.replace("__ISIN_MAP__", _ISIN_MAP)
    body = body.replace("__JS__", _JS)
    return HTMLResponse(body)


@app.get("/select", response_class=HTMLResponse)
async def selection_page():
    """Serve the S&P 100 selection UI."""
    body = _SELECT_HTML.replace("__CSS__", f"<style>\n{_CSS}\n</style>")
    body = body.replace("__JS__", _SELECT_JS)
    return HTMLResponse(body)


@app.get("/jobs/{job_id}")
def get_progress_job(job_id: str, after: int = Query(-1, ge=-1)):
    """Return a job snapshot and progress events newer than the given sequence."""
    job = progress_jobs.get(job_id, after=after)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Progress job not found or expired."})
    return job


def _start_progress_job(operation: str, run: Callable[[], dict]) -> dict | JSONResponse:
    job = progress_jobs.start(operation, run)
    return job if job is not None else _busy_response()


@app.post("/select/run")
def select_run_endpoint(
    stage: str = Form("all"),
    selection_top_n: str = Form("10"),
    selection_recursion_limit: str = Form("160"),
    selection_verdict_concurrency: str = Form("2"),
    force_refresh: str = Form("off"),
    progress: str = Form("off"),
):
    """Run the S&P 100 selection funnel, optionally returning a live progress job."""
    try:
        from models.selection import run_selection

        cfg = Config.from_dict({
            "selection_top_n": selection_top_n,
            "selection_recursion_limit": selection_recursion_limit,
            "selection_verdict_concurrency": selection_verdict_concurrency,
        })
        run = lambda: run_selection(
            cfg,
            stage=stage,
            force_refresh=force_refresh in ("on", "true", "1"),
        )
        if progress in ("on", "true", "1"):
            return _start_progress_job("selection", run)
        if not progress_jobs.try_acquire_slot():
            return _busy_response()
        try:
            return run()
        finally:
            progress_jobs.release_slot()
    except Exception as exc:
        return _error_response(exc, "Selection")


def _run_optimization(cfg: Config) -> dict:
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
    progress: str = Form("off"),
):
    """Run optimization, optionally returning a job ID for live progress polling."""
    try:
        cfg = Config.from_dict({
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
        })
        if progress in ("on", "true", "1"):
            return _start_progress_job("optimization", lambda: _run_optimization(cfg))
        if not progress_jobs.try_acquire_slot():
            return _busy_response()
        try:
            return _run_optimization(cfg)
        finally:
            progress_jobs.release_slot()
    except Exception as exc:
        return _error_response(exc, "Optimization")
