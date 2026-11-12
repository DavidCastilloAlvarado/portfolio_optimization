"""Tests for the FastAPI endpoints (data layer mocked)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from core import pipeline
from web.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _raw_data(n_days=60):
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    return pd.DataFrame({
        "Date": dates,
        "AAA": 100 * (1.01) ** np.arange(n_days),
        "BBB": 100 * (1.005) ** np.arange(n_days),
    })


@pytest.fixture
def mock_data(monkeypatch):
    monkeypatch.setattr(pipeline, "bulk_stocks", lambda shares, days: _raw_data())
    monkeypatch.setattr(pipeline, "run_backtest", lambda *a, **k: None)


def test_index_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in res.headers["content-security-policy"]
    assert "Portfolio Optimizer" in res.text
    assert "per-ticker-panel" in res.text
    assert "<style>" in res.text
    assert "window.ISIN_TICKER_MAP" in res.text
    assert 'id="job-progress-events"' in res.text
    assert "runProgressJob" in res.text
    assert "IE00BFMXXD54" in res.text
    assert "VUAA" in res.text


def test_optimize_success(client, mock_data):
    res = client.post(
        "/optimize",
        data={
            "resample": "none",
            "days": "60",
            "shares": "AAA,BBB",
            "w_limits": "0.02,0.9",
            "min_variance": "off",
            "monto_usd": "10000",
            "monthly_delta": "0",
            "sim_days": "30",
            "risk_free_annual_perc": "5",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Mean-Variance" in data["strategy"]
    assert [w["ticker"] for w in data["weights"]] == ["AAA", "BBB"]
    assert abs(sum(w["weight_pct"] for w in data["weights"]) - 100.0) < 1e-6
    assert data["portfolio_return_pct"] > 0
    assert "backtest" not in data


def test_optimize_isin_shares_include_symbol(client, monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=60)
    isin = "IE00BFMXXD54"
    monkeypatch.setattr(pipeline, "bulk_stocks", lambda shares, days: pd.DataFrame({
        "Date": dates,
        isin: 100 * (1.01) ** np.arange(60),
        "BBB": 100 * (1.005) ** np.arange(60),
    }))
    monkeypatch.setattr(pipeline, "run_backtest", lambda *a, **k: None)
    res = client.post(
        "/optimize",
        data={"shares": f"{isin},BBB", "w_limits": "0.02,0.9", "monthly_delta": "0"},
    )
    assert res.status_code == 200
    weights = {w["ticker"]: w for w in res.json()["weights"]}
    assert weights[isin]["symbol"] == "VUAA"
    assert weights["BBB"]["symbol"] is None


def test_optimize_min_variance_with_per_ticker(client, mock_data):
    res = client.post(
        "/optimize",
        data={
            "shares": "AAA,BBB",
            "w_limits": "0.02,0.9",
            "w_limits_per_ticker": "AAA:0.6,0.9",
            "min_variance": "on",
            "monthly_delta": "0",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Minimal Variance" in data["strategy"]
    weights = {w["ticker"]: w["weight_pct"] for w in data["weights"]}
    assert 60.0 - 1e-4 <= weights["AAA"] <= 90.0 + 1e-4


def test_optimize_kelly_growth(client, mock_data):
    res = client.post(
        "/optimize",
        data={
            "shares": "AAA,BBB",
            "w_limits": "0.02,0.9",
            "strategy": "kelly",
            "kelly_fraction": "0.5",
            "monthly_delta": "0",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Growth" in data["strategy"]
    assert "Kelly" in data["strategy"]
    assert [w["ticker"] for w in data["weights"]] == ["AAA", "BBB"]
    assert abs(sum(w["weight_pct"] for w in data["weights"]) - 100.0) < 1e-6
    assert "growth_rate_pct" in data


def test_optimize_llm_bl(client, mock_data, monkeypatch):
    def fake_run_blm(returns, names, cov, cfg):
        return {
            "posterior_mean": np.array([0.0015, 0.0005]),
            "prior_mean": np.array([0.0007, 0.0003]),
            "views": {
                "AAA": {"q": 0.0015, "omega": 1e-6, "samples": [0.0015], "n_ok": 1},
            },
            "view_names": ["AAA"],
            "from_cache": True,
            "degraded": False,
            "model": "test-model",
        }

    monkeypatch.setattr(pipeline, "run_blm", fake_run_blm)
    res = client.post(
        "/optimize",
        data={
            "shares": "AAA,BBB",
            "w_limits": "0.02,0.9",
            "strategy": "llm_bl",
            "llm_samples": "3",
            "llm_tau": "0.05",
            "blm_prior": "inverse_vol",
            "blm_horizon_days": "7",
            "llm_view_engine": "agent",
            "llm_reasoning_effort": "high",
            "llm_agent_websearch": "off",
            "llm_agent_max_tokens": "2048",
            "llm_agent_recursion_limit": "60",
            "monthly_delta": "0",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Black-Litterman" in data["strategy"]
    assert "LLM views" in data["strategy"]
    lv = data["llm_views"]
    assert lv["samples_per_asset"] == 3
    assert lv["tau"] == pytest.approx(0.05)
    assert lv["horizon_days"] == 7
    assert lv["from_cache"] is True
    assert lv["view_engine"] == "direct"
    assert [a["ticker"] for a in lv["assets"]] == ["AAA"]


def test_optimize_error_returns_400(client, mock_data, monkeypatch):
    def boom(shares, days):
        raise ValueError("No price data returned for ticker 'AAA'")

    monkeypatch.setattr(pipeline, "bulk_stocks", boom)
    res = client.post("/optimize", data={"shares": "AAA"})
    assert res.status_code == 400
    body = res.json()
    assert "No price data" in body["error"]
    assert "request_id" in body
    assert "traceback" not in body


def test_select_page_serves_ui(client):
    res = client.get("/select")
    assert res.status_code == 200
    assert "S&amp;P 100 Selection" in res.text
    assert 'href="/" class="nav-optimizer"' in res.text
    assert 'href="/select" class="nav-selection active" aria-current="page"' in res.text
    assert "selection_recursion_limit" in res.text
    assert "selection_verdict_concurrency" in res.text
    assert "<style>" in res.text
    assert "btn-stage-all" in res.text
    assert 'id="job-progress-events"' in res.text
    assert "runProgressJob" in res.text


def test_index_has_page_nav(client):
    res = client.get("/")
    assert 'href="/select" class="nav-selection"' in res.text
    assert 'class="nav-optimizer active" aria-current="page">Optimizer' in res.text


def _assert_js_not_leaked(html: str, markers: list[str]) -> None:
    import re

    stripped = re.sub(r"<script\b.*?</script>", "", html, flags=re.S)
    for marker in markers:
        assert marker not in stripped, f"raw JS leaked into page body: {marker!r}"
    assert html.count("<script") == html.count("</script>")


def test_select_page_js_is_script_wrapped(client):
    _assert_js_not_leaked(client.get("/select").text, ["applyTheme", "runStage", "escapeHtml"])


def test_index_page_js_is_script_wrapped(client):
    _assert_js_not_leaked(client.get("/").text, ["buildPerTickerRows"])


def test_select_run_stage_screen(client, monkeypatch):
    import models.selection as sel
    captured = {}

    def fake_run(cfg, stage="all", settings=None, mcp_settings=None, force_refresh=False):
        captured["stage"] = stage
        captured["top_n"] = cfg.selection_top_n
        captured["recursion_limit"] = cfg.selection_recursion_limit
        captured["concurrency"] = cfg.selection_verdict_concurrency
        captured["force"] = force_refresh
        return {"stage": stage, "as_of": "2026-01-01", "screen": {"sectors": [], "names": []},
                "briefing": None, "sectors": [], "selection": [], "vetoed": [],
                "from_cache": True, "degraded": {}}

    monkeypatch.setattr(sel, "run_selection", fake_run)
    res = client.post(
        "/select/run",
        data={
            "stage": "screen",
            "selection_top_n": "12",
            "selection_recursion_limit": "180",
            "selection_verdict_concurrency": "3",
            "force_refresh": "off",
        },
    )
    assert res.status_code == 200
    assert res.json()["stage"] == "screen"
    assert captured["top_n"] == 12
    assert captured["recursion_limit"] == 180
    assert captured["concurrency"] == 3
    assert captured["force"] is False


def test_select_run_force_refresh_and_defaults(client, monkeypatch):
    import models.selection as sel
    captured = {}

    def fake_run(cfg, stage="all", settings=None, mcp_settings=None, force_refresh=False):
        captured["stage"] = stage
        captured["force"] = force_refresh
        return {"stage": stage, "as_of": "x", "screen": {}, "briefing": None,
                "sectors": [], "selection": [], "vetoed": [], "from_cache": False, "degraded": {}}

    monkeypatch.setattr(sel, "run_selection", fake_run)
    res = client.post("/select/run", data={"stage": "all", "force_refresh": "on"})
    assert res.status_code == 200
    assert captured["stage"] == "all"
    assert captured["force"] is True


def test_select_run_error_returns_400(client, monkeypatch):
    import models.selection as sel

    def boom(*a, **k):
        raise ValueError("selection universe missing")

    monkeypatch.setattr(sel, "run_selection", boom)
    res = client.post("/select/run", data={"stage": "screen"})
    assert res.status_code == 400
    body = res.json()
    assert "selection universe missing" in body["error"]
    assert "request_id" in body
    assert "traceback" not in body


def _wait_for_progress_job(client, job_id):
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail(f"progress job {job_id} did not finish")


def test_optimize_progress_job_returns_result(client, mock_data):
    response = client.post(
        "/optimize",
        data={
            "shares": "AAA,BBB",
            "w_limits": "0.02,0.9",
            "monthly_delta": "0",
            "progress": "on",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = _wait_for_progress_job(client, job_id)

    assert job["status"] == "completed"
    assert [weight["ticker"] for weight in job["result"]["weights"]] == ["AAA", "BBB"]
    assert any(event["tag"] == "pipeline" for event in job["events"])


def test_selection_progress_job_captures_incremental_events(client, monkeypatch):
    import models.selection as sel
    from core.progress import progress_log

    def fake_run(cfg, stage="all", settings=None, mcp_settings=None, force_refresh=False):
        progress_log("briefing", "research phase started")
        return {
            "stage": stage,
            "as_of": "2026-01-01",
            "screen": {"sectors": [], "names": []},
            "briefing": None,
            "sectors": [],
            "selection": [],
            "vetoed": [],
            "from_cache": False,
            "degraded": {},
        }

    monkeypatch.setattr(sel, "run_selection", fake_run)
    response = client.post("/select/run", data={"stage": "all", "progress": "on"})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = _wait_for_progress_job(client, job_id)

    assert job["status"] == "completed"
    assert job["result"]["stage"] == "all"
    assert any(event["message"] == "research phase started" for event in job["events"])
    incremental = client.get(f"/jobs/{job_id}", params={"after": 0}).json()
    assert all(event["sequence"] > 0 for event in incremental["events"])


def test_progress_job_reports_failure(client, monkeypatch):
    import models.selection as sel

    def boom(*a, **k):
        raise ValueError("selection calculation failed")

    monkeypatch.setattr(sel, "run_selection", boom)
    response = client.post("/select/run", data={"stage": "screen", "progress": "on"})
    assert response.status_code == 200
    job = _wait_for_progress_job(client, response.json()["job_id"])

    assert job["status"] == "failed"
    assert job["error"] == "selection calculation failed"
    assert any(event["message"].startswith("Failed:") for event in job["events"])
