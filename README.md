# Portfolio Optimization Engine

A Python tool that finds the optimal asset allocation for your investment portfolio using modern portfolio theory — balancing **risk** and **return** through optimization and backtesting.

## WebUI

![alt text](<img.png>)

## Features

- **Mean-Variance (Sharpe)** — maximizes risk-adjusted return
- **Minimal Variance** — minimizes portfolio risk
- **Historical backtest** — lump-sum and DCA strategies
- **Web UI** — configure everything via browser, no CLI needed
- **Multi-source data** — Yahoo Finance (tickers) with JustETF fallback (ISINs), automatic fetch with CSV caching

## Project Structure

```
portfolio_optimization/
├── main.py                 # CLI entry point
├── pyproject.toml          # Dependencies + poe tasks
├── README.md
├── config/
│   ├── __init__.py
│   └── defaults.py         # All parameters as a dataclass
├── core/                   # Domain layer
│   ├── pipeline.py         # Shared data → optimize → backtest flow
│   ├── data/
│   │   ├── cache.py        # Daily CSV cache in temp/
│   │   └── loader.py       # Yahoo Finance fetch + return preparation
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
│   ├── main.py             # FastAPI web UI
│   ├── renderer.py         # Result rendering helpers
│   └── static/
│       ├── index.html      # HTML template
│       ├── styles.css      # Coder-style theme (dark default + light)
│       └── app.js          # Client-side logic
└── tests/
    ├── conftest.py         # Shared fixtures
    ├── test_portfolio.py   # Portfolio math
    ├── test_solvers.py     # Solvers + bounds integration
    ├── test_bounds.py      # Weight-bound resolution
    ├── test_cache.py       # CSV cache
    ├── test_loader.py      # Fetch + return preparation (mocked HTTP)
    ├── test_backtest.py    # Backtest engine (synthetic prices)
    ├── test_pipeline.py    # End-to-end pipeline (mocked data)
    ├── test_config.py      # Config dataclass
    └── test_web_api.py     # FastAPI endpoints (mocked data)
```

## Quickstart

### Install dependencies

```bash
poetry install
```

### Run the web UI (recommended)

```bash
poetry run poe web
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Run CLI

```bash
poetry run poe opt
```

### Run tests

```bash
poetry run poe test
```

## Configuration

All parameters are defined in `config/defaults.py`. The default configuration:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resample` | `"none"` | Data frequency: `"none"`, `"week"`, `"month"` |
| `days` | `720` | Calendar days of historical data |
| `shares` | `['XLU', 'QQQ', ...]` | Ticker symbols and ISINs |
| `w_limits` | `(0.02, 0.12)` | Min/max weight per asset |
| `min_variance` | `False` | `True` = Min-Variance, `False` = Max Sharpe |
| `monto_usd` | `10000` | Initial investment (USD) |
| `monthly_delta` | `300` | Monthly DCA contribution (USD) |
| `sim_days` | `252` | Trading days for simulation |
| `risk_free_annual_perc` | `5` | Annual risk-free rate (%) |

## Data Sources

Daily close prices come from two sources, tried in order:

1. **Yahoo Finance** (primary) — [Query v8 API](https://github.com/LearnCash/finance-yahoo), looked up by ticker symbol (e.g. `SPY`, `AAPL`).
2. **JustETF** (fallback) — [JustETF performance-chart API](https://www.justetf.com), looked up by ISIN (e.g. `IE00BFMXXD54`). Used automatically when Yahoo fails **and** the symbol matches the ISIN pattern: 12 characters — 2-letter country code, 9 alphanumeric characters, and a numeric check digit (`^[A-Z]{2}[A-Z0-9]{9}[0-9]$`). Plain tickers never fall back.

Both sources are cached daily under `temp/` (`{SYMBOL}_{YYYY-MM-DD}.csv`), so each symbol is fetched at most once per day regardless of source.

## How It Works

### 1. Data Loading

Fetches daily close prices from Yahoo Finance (primary) with an automatic JustETF fallback for ISINs — see [Data Sources](#data-sources). Results are cached locally under `temp/` to avoid repeated network calls.

Computes percentage returns (daily, weekly, or monthly) with time-weighted interpolation for missing values.

### 2. Optimization

Two strategies from `optimizer.py`:

- **Mean-Variance (Sharpe)** — Uses `scipy.optimize.minimize` (SLSQP) to maximize the Sharpe ratio by minimizing its inverse. Based on [Harry Markowitz's Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory) (1952).

- **Minimal Variance** — Uses `cvxopt.qp` (quadratic programming) to find the lowest possible variance portfolio subject to weight constraints.

The Sharpe ratio, `(portfolio_return - risk_free_rate) / portfolio_std`, was introduced by [William F. Sharpe](https://en.wikipedia.org/wiki/Sharpe_ratio) (1966).

### 3. Backtest

Simulates portfolio performance using real historical price data under two approaches:

- **Lump-sum** — Full initial investment upfront
- **DCA** — Initial investment + monthly contributions every ~21 trading days

Reports: final value, total return %, max drawdown, annualized Sharpe ratio.

## Dependencies

| Library | Purpose |
|---------|---------|
| [pandas](https://pandas.pydata.org/) | Data manipulation and time-series |
| [numpy](https://numpy.org/) | Numerical computing |
| [scipy](https://scipy.org/) | SLSQP optimization |
| [cvxopt](https://cvxopt.org/) | Quadratic programming |
| [requests](https://requests.readthedocs.io/) | HTTP client for Yahoo Finance and JustETF |
| [tqdm](https://tqdm.github.io/) | Progress bars |
| [fastapi](https://fastapi.tiangolo.com/) | Web API framework |
| [uvicorn](https://www.uvicorn.org/) | ASGI server |

## Key Concepts

| Concept | Reference |
|---------|-----------|
| Modern Portfolio Theory (Markowitz, 1952) | [Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory) · [Paper](https://www.aeaweb.org/articles?id=10.1257/aer.40.3.77) |
| Sharpe Ratio (Sharpe, 1966) | [Wikipedia](https://en.wikipedia.org/wiki/Sharpe_ratio) |
| Minimal Variance Portfolio | [Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory#Minimum_variance_portfolio) |
| Dollar-Cost Averaging | [Investopedia](https://www.investopedia.com/terms/d/dollarcostaveraging.asp) |

