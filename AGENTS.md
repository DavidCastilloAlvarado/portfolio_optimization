# AGENTS.md — Portfolio Optimization Engine

## Project Overview

A Python portfolio optimization engine using Modern Portfolio Theory with Mean-Variance (Sharpe) and Minimal Variance strategies. Includes a FastAPI web UI, multi-source market data (Yahoo Finance primary, JustETF fallback for ISINs) with CSV caching, and historical backtesting (lump-sum + DCA).

## Tech Stack

- **Language**: Python 3.10+
- **Package Manager**: Poetry
- **Optimization**: `scipy.optimize.minimize` (SLSQP), `cvxopt.qp`
- **Web Framework**: FastAPI + Uvicorn
- **Data**: pandas, numpy
- **Testing**: pytest
- **Task Runner**: poe (poethepoet)
- **External APIs**: Yahoo Finance Query v8, JustETF (performance-chart API)

## Project Structure

```
portfolio_optimization/
├── main.py                 # CLI entry point
├── pyproject.toml          # Dependencies + poe tasks
├── README.md
├── doc/                    # Reference docs (gitignored), e.g. etfs.md
├── config/
│   ├── __init__.py
│   └── defaults.py         # Config dataclass (all tunable parameters)
├── core/                   # Domain layer
│   ├── pipeline.py         # Shared data → optimize → backtest flow (run_pipeline)
│   ├── data/
│   │   ├── cache.py        # Daily CSV cache in temp/
│   │   └── loader.py       # Yahoo fetch + JustETF fallback, return preparation
│   ├── optimization/
│   │   ├── portfolio.py    # Portfolio mean / variance math
│   │   ├── bounds.py       # Weight-bound resolution (global + per-ticker)
│   │   └── solvers.py      # Mean-Variance + Min-Variance solvers
│   └── simulation/
│       └── backtest.py     # Lump-sum + DCA backtest
├── models/
│   ├── __init__.py
│   └── blitterman.py       # Black-Litterman + market cap utilities
├── web/                    # Presentation layer
│   ├── __init__.py
│   ├── main.py             # FastAPI endpoints
│   ├── renderer.py         # Result building helpers
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
└── tests/
    ├── __init__.py
    ├── conftest.py         # Shared fixtures
    ├── test_portfolio.py   # Portfolio math
    ├── test_solvers.py     # Solvers
    ├── test_bounds.py      # Weight-bound resolution
    ├── test_cache.py       # CSV cache
    ├── test_loader.py      # Fetch + fallback + return preparation (mocked HTTP)
    ├── test_backtest.py    # Backtest engine
    ├── test_pipeline.py    # End-to-end pipeline (mocked data)
    ├── test_config.py      # Config dataclass
    └── test_web_api.py     # FastAPI endpoints (mocked data)
```

## Commands

```bash
# Install dependencies
poetry install

# Run CLI
poetry run poe opt

# Run web UI
poetry run poe web

# Run tests
poetry run poe test
```

## Development Conventions

### Code Style
- No comments unless explicitly asked
- Use type hints throughout
- Follow existing code style (no trailing whitespace, consistent spacing)
- Use `float`/`int`/`str`/`list`/`dict` built-in generics (Python 3.10+)

### Configuration
- All parameters live in `config/defaults.py` (`Config` dataclass)
- Never hardcode values — use `Config` or `Config.from_dict()` for form data
- Default values in `Config` dataclass fields

### Data Loading
- **Primary source**: Yahoo Finance Query v8, looked up by ticker symbol
- **Fallback source**: JustETF performance-chart API (`core/data/loader.py::fetch_prices_justetf`), looked up by ISIN
- Fallback triggers only when Yahoo fails **and** the symbol matches the ISIN pattern `^[A-Z]{2}[A-Z0-9]{9}[0-9]$` (`is_isin()`); plain tickers never fall back
- Data cached daily in `temp/` as `{SYMBOL}_{YYYY-MM-DD}.csv` (both sources, shared cache)
- `temp/` CSVs are gitignored (`*.csv` in `.gitignore`)
- **Date convention**: every table returned by `load_table` has a `Date` column of `datetime64[ns, UTC]` normalized to midnight (`.dt.normalize()`). Yahoo timestamps carry the US market-open time (13:30 UTC) while JustETF uses midnight — never drop the normalization, or mixed-source merges/backtests silently break
- `read_prices()` (cache → backtest) applies the same UTC midnight normalization
- `prepare_returns()` handles resampling (daily/weekly/monthly)

### Optimization
- `core/optimization/solvers.py` exports `optimize()` as the main entry point
- Two strategies: Mean-Variance (Sharpe) and Minimal Variance
- `build_asset_bounds()` (in `bounds.py`) merges global `w_limits` with `w_limits_per_ticker`
- Weights always sum to 1.0
- `w_limits` tuple: `(min_weight, max_weight)` per asset

### Web UI
- `web/main.py` — FastAPI app with `/` (HTML) and `/optimize` (POST) endpoints
- `web/renderer.py` — builds result dicts from optimizer output
- Static files injected inline (HTML replaces `__CSS__` and `__JS__` placeholders)
- Form data parsed via `Config.from_dict()`

### Simulation/Backtest
- `core/simulation/backtest.py::run_backtest` uses real historical prices from the CSV cache (via `read_prices`), not random data
- Two modes: lump-sum and DCA (monthly contributions every 21 trading days)
- Returns dict with `lump_sum` and/or `dca` keys (or `error` when not enough data)
- Stats: final value, total return %, max drawdown, annualized Sharpe

### Testing
- `pytest tests/ -v` runs all tests
- Network is always mocked (`requests.get` monkeypatched); cache tests use `tmp_path` via a monkeypatched `CACHE_DIR`
- Use `pytest.approx()` for floating-point comparisons
- Regression tests exist for: ISIN matching, JustETF parsing, fallback behavior, mixed-source (cached + fresh, Yahoo + JustETF) merges, and backtest index alignment

### Error Handling
- API errors return `JSONResponse(status_code=400)` with `error` + `traceback`
- Data loading errors raise `ValueError` with descriptive messages
- Optimization failures raise `RuntimeError` from scipy/cvxopt

## Key Files Reference

| File | Purpose |
|------|---------|
| `config/defaults.py` | Central config — modify defaults here |
| `core/pipeline.py` | Shared computation flow (CLI + web) |
| `core/data/loader.py` | Yahoo + JustETF fetch, fallback logic, returns |
| `core/data/cache.py` | CSV cache read/write |
| `core/optimization/solvers.py` | Core optimization logic |
| `core/simulation/backtest.py` | Backtest engine |
| `web/main.py` | FastAPI server |
| `web/renderer.py` | Result formatting |
| `tests/test_loader.py` | Data source + fallback tests |

## Risk-Free Rate Calculation

Both the optimizer and backtest use the 252-trading-day basis (consistent with the display annualization in `web/renderer.py`):

- Optimizer (daily): `(1 + annual_perc/100) ** (1/252) - 1` — `Config.risk_free`
- Backtest (daily): `(1 + annual_rate) ** (1/252) - 1`

## Backtest Stats

- **Sharpe (annualized)**: `(mean_daily_return - daily_rf) / std_daily_return * sqrt(252)`
- **Max Drawdown**: `(cummax - current) / cummax * 100`
- **Total Return %**: `(final_value - total_invested) / total_invested * 100`
