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
    assert "Portfolio Optimizer" in res.text
    assert "per-ticker-panel" in res.text
    assert "<style>" in res.text


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


def test_optimize_error_returns_400(client, mock_data, monkeypatch):
    def boom(shares, days):
        raise ValueError("No price data returned for ticker 'AAA'")

    monkeypatch.setattr(pipeline, "bulk_stocks", boom)
    res = client.post("/optimize", data={"shares": "AAA"})
    assert res.status_code == 400
    body = res.json()
    assert "No price data" in body["error"]
    assert "traceback" in body
