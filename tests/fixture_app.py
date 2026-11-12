"""Hermetic FastAPI app for browser UI tests with synthetic market data.

Selection uses deterministic agent contracts; Black-Litterman uses the local
OpenAI-compatible mock started by scripts/verify_ui.py.
"""
import numpy as np
import pandas as pd

import core.pipeline as pipeline
import core.selection as selection_data
import core.simulation.backtest as backtest
import models.selection as selection_pipeline
from models.selection_agents import MarketBriefing, SectorView, SelectionVerdict
from web.main import app

FIXTURE_DAYS = 720


def _fixture_prices(ticker: str) -> pd.Series:
    seed = sum(ord(c) for c in ticker.upper()) * 7919
    rng = np.random.default_rng(seed)
    drift = 0.0001 + (seed % 100) / 100 * 0.0004
    vol = 0.008 + (seed % 5) * 0.004
    index = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").normalize(), periods=FIXTURE_DAYS, freq="B"
    )
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, FIXTURE_DAYS)))
    return pd.Series(close, index=index, name=ticker)


def _fake_bulk_stocks(shares: list, days_back: int) -> pd.DataFrame:
    data = pd.DataFrame({"Date": _fixture_prices(shares[0]).index})
    for share in shares:
        data[share] = _fixture_prices(share).values
    return data.tail(days_back)


def _fake_read_prices(ticker: str) -> pd.Series:
    return _fixture_prices(ticker)


def _fixture_selection_screen(cfg, force_refresh=False) -> dict:
    sector_names = [
        "Information Technology", "Health Care", "Financials", "Consumer Discretionary",
        "Communication Services", "Industrials", "Consumer Staples", "Energy",
        "Utilities", "Materials", "Real Estate",
    ]
    sectors = [
        {
            "sector": sector,
            "n_members": 10 - index % 4,
            "quant_score": round(0.95 - index * 0.055, 4),
            "momentum": round(0.9 - index * 0.04, 4),
            "breadth": round(0.88 - index * 0.035, 4),
        }
        for index, sector in enumerate(sector_names)
    ]
    names = []
    for index in range(12):
        ticker = f"MOCK{index + 1}"
        sector = sector_names[index % 3]
        names.append({
            "ticker": ticker,
            "name": f"Mock Company {index + 1}",
            "sector": sector,
            "deduped": False,
            "finance_score": round(0.9 - index * 0.025, 4),
            "momentum": 0.8 - index * 0.02,
            "quality": 0.75,
            "trailing_pe": 18.0 + index,
            "revenue_growth": 0.08,
            "margins": 0.2,
            "metrics": {
                "ret_1m": 0.04 + index * 0.002,
                "ret_3m": 0.1 + index * 0.003,
                "ret_6m": 0.17 + index * 0.004,
                "off_high_52w": -0.03,
                "max_dd_3m": -0.08,
                "vol_3m": 0.18,
                "above_ma50": True,
                "rel_volume": 1.1,
            },
        })
    return {
        "as_of": pd.Timestamp.now(tz="UTC").date().isoformat(),
        "sectors": sectors,
        "names": names,
        "from_cache": False,
        "skipped": [],
    }


def _fixture_market_briefing(candidate_sectors, cfg, **kwargs) -> MarketBriefing:
    return MarketBriefing(
        regime_summary="Mock market conditions are broadly risk-on with a balanced outlook.",
        macro_risks=["Mock inflation and rates risk"],
        trending_themes=["Mock earnings growth"],
        overall_market_risk=0.3,
        sectors=[
            SectorView(
                sector=sector,
                trend_score=0.78,
                regime="sustained",
                driver="Mock research indicates broad, sustained momentum.",
                fragility=0.15,
                conviction=0.82,
            )
            for sector in candidate_sectors
        ],
    )


def _fixture_name_verdicts(candidates, cfg, **kwargs) -> dict[str, SelectionVerdict]:
    return {
        candidate["ticker"]: SelectionVerdict(
            include=True,
            conviction=0.84,
            thesis="Mock fundamentals support the recent price trend.",
            catalysts=["Mock product launch"],
            overhangs=["Mock valuation risk"],
            earnings_in_days=30,
        )
        for candidate in candidates
    }


pipeline.bulk_stocks = _fake_bulk_stocks
backtest.read_prices = _fake_read_prices
selection_data.run_screen = _fixture_selection_screen
selection_pipeline.run_market_briefing = _fixture_market_briefing
selection_pipeline.run_name_verdicts = _fixture_name_verdicts
