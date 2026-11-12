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
- **S&P 100 Selection** — quant funnel + LLM research agents that decide *which* tickers deserve a seat in the portfolio (separate page, feeds the optimizer)

## Project Structure

```
portfolio_optimization/
├── main.py                 # CLI entry point (optimizer)
├── select.py               # CLI entry point (S&P 100 selection)
├── pyproject.toml          # Dependencies + poe tasks
├── README.md
├── config/
│   ├── __init__.py
│   ├── defaults.py         # All parameters as a dataclass
│   ├── isin_ticker_map.json  # ISIN → display ticker map (edit freely, no code)
│   └── sp100.json          # S&P 100 universe (ticker → name/sector/sub-industry, edit freely)
├── core/                   # Domain layer
│   ├── pipeline.py         # Shared data → optimize → backtest flow
│   ├── selection.py        # S&P 100 quant funnel (screen, scoring, sectors, dedup)
│   ├── data/
│   │   ├── cache.py        # Daily CSV cache in temp/
│   │   ├── loader.py       # Yahoo Finance fetch + return preparation
│   │   └── quotes.py       # Yahoo batch quotes (crumb session) + v10 fundamentals
│   ├── optimization/
│   │   ├── portfolio.py    # Portfolio mean / variance math
│   │   ├── bounds.py       # Weight-bound resolution (global + per-ticker)
│   │   └── solvers.py      # Mean-Variance + Min-Variance solvers
│   └── simulation/
│       └── backtest.py     # Lump-sum + DCA backtest
├── models/
│   ├── __init__.py
│   ├── blitterman.py       # Black-Litterman + market cap utilities
│   ├── selection_agents.py # Selection LLM agents (market briefing + name verdicts)
│   └── selection.py        # Selection orchestrator (blend + hard gates + degradation)
├── web/                    # Presentation layer
│   ├── __init__.py
│   ├── main.py             # FastAPI web UI (optimizer + selection pages)
│   ├── renderer.py         # Result rendering helpers
│   └── static/
│       ├── index.html      # Optimizer page
│       ├── select.html     # S&P 100 selection page
│       ├── styles.css      # Coder-style theme (dark default + light)
│       ├── app.js          # Optimizer client-side logic
│       └── select.js       # Selection client-side logic
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
    ├── test_web_api.py     # FastAPI endpoints (mocked data)
    ├── test_selection.py   # Quant funnel + orchestrator (mocked data/agents)
    ├── test_market_agent.py# Selection agents (mocked LLM + MCP)
    └── test_quotes.py      # Yahoo crumb quotes + fundamentals (mocked HTTP)
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

Open [http://localhost:8000](http://localhost:8000) in your browser. Optimization and Selection runs show their current phase, recent progress events, and elapsed time in the page; progress jobs are kept in the web process's memory and are cleared when it restarts.

### Run CLI

```bash
poetry run poe opt
```

### Run the S&P 100 selection

```bash
poetry run poe select                # full funnel (quant + LLM agents)
poetry run poe select --stage screen # quant screen only (fast, no LLM)
```

or use the dedicated **Selection** page at [http://localhost:8000/select](http://localhost:8000/select) (staged buttons + "Send to Optimizer").

### Run tests

```bash
poetry run poe test
```

### Verify the web UI (Playwright)

Mandatory after any change to `web/static/*` or the page endpoints — drives a real browser, checks both pages for leaked script/CSS, missing elements, and uncaught JS errors, renders the Selection screen and full funnel, and exercises every optimizer strategy. The verifier uses synthetic market data, fixture Selection agents, and a local mock LLM, so it does not call external services or a live model. Inspect the screenshots it creates in `temp/ui_verify_*.png`.

```bash
poetry run poe verify-ui
```

For manual local development, start the deterministic OpenAI-compatible mock endpoint with `poetry run poe mock-llm` (default: `http://127.0.0.1:8001`). Configure the app to use its direct Black-Litterman path with:

```bash
LLM_BASE_URL=http://127.0.0.1:8001/v1 LLM_API_KEY=mock LLM_MODEL=portfolio-mock poetry run poe web
```

In the optimizer form, select the **Direct (history only)** view engine; the mock does not provide web-research tools for agent mode. The UI verifier configures this automatically and uses deterministic fixture results for Selection's agents. The mock returns synthetic forecasts for testing only; do not use its output to make investment decisions.

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

## ISIN → Ticker Map

When you use ISINs, every view shows the mapped display ticker next to the reference ISIN — the web results table, the per-ticker weight panel, the CLI output, and the raw JSON (e.g. `IE00BFMXXD54` shows as `VUAA` + `IE00BFMXXD54`).

**The map lives in a single file: [`config/isin_ticker_map.json`](config/isin_ticker_map.json).**

It is plain JSON — **no code changes are needed to add or edit entries**. One mapping per line, `"ISIN": "TICKER"`:

```json
{
  "IE00BFMXXD54": "VUAA",
  "IE00B53SZB19": "CNDX",
  "IE00B4ND3602": "IGLN"
}
```

To add a new ISIN, open the file and insert one more line (watch the trailing commas):

```json
{
  "IE00BFMXXD54": "VUAA",
  "IE00B53SZB19": "CNDX",
  "IE00XXXXXXXX": "TICK"
}
```

Notes:

- ISINs not in the map simply show as the bare ISIN (no ticker) — nothing breaks.
- The map is **display-only**: data is still fetched by ticker (Yahoo) or ISIN (JustETF).
- Invalid JSON stops the app at startup with a message pointing at the exact error in the file, so typos are easy to fix.

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

## S&P 100 Selection — which names to include

The optimizer decides **how much** of each asset to hold; the selection feature decides **which** tickers deserve a seat in the first place. It screens the S&P 100 (top 100 S&P 500 constituents by market cap, listed in `config/sp100.json`) and blends quantitative momentum with LLM web research. The feature is additive: the optimizer page is unchanged, and the selection output ("Send to Optimizer") simply pre-fills its asset list.

### Funnel

1. **Quant screen** (`core/selection.py`) — per name: 1m/3m/6m returns, distance to 52w high, 3m max drawdown, volatility, relative volume + quote fundamentals (P/E, revenue growth, margins). `finance = 0.75·momentum + 0.25·quality` (quality = 0 without positive earnings). Sectors aggregate member momentum (60%) + breadth (40%). Near-duplicate twins (126d return correlation ≥ 0.95 in the same sub-industry, e.g. GOOG/GOOGL) keep only the larger cap.
2. **Market briefing agent** (`models/selection_agents.py`) — a LangChain research agent (same protocol as the Black-Litterman agent) that runs 10–18 web searches across macro (rates, inflation, labor), geopolitics (war, sanctions, energy), the technology/AI industry, each candidate sector, and market sentiment, writes a daily memo, and returns a regime call per sector (`trend_score`, `fragility`, `conviction`).
3. **Name verdict agents** — one agent pass per candidate name. They reuse the *shared* per-ticker research memo of the BL agent (`temp/research_{TICKER}_{date}.md` — a memo written by either feature is used by the other), do targeted searches only when no memo exists, and return a verdict: `include`, `conviction`, thesis, catalysts, overhangs, distance to next earnings. Independent name agents run with configurable concurrency (default 2; reduce it for smaller local models).

The Selection page exposes the briefing's LangGraph step limit (default 160; raise it if the market briefing exhausts its tool budget) and the name-agent concurrency (1–8). Increasing concurrency speeds up the per-name phase, not the single market-briefing agent, and increases simultaneous LLM/MCP load.

### Blend + hard gates

```
sector_final = 0.6 · quant_score + 0.4 · trend_score · (1 − 0.5 · fragility)
name_final   = 0.5 · finance_score + 0.5 · conviction
```

(no score can override): a name's sector must be in the top 3 blended sectors; an LLM `include=false` vetoes the name; earnings inside 5 trading days flag the name (kept, marked). Weights and the flag window are `Config` parameters (`selection_*`).

### Degradation

Every LLM stage is optional: briefing fails → sectors stay quant-only (flagged in the UI/CLI); a verdict fails → the name stays quant-only; MCP or LLM not configured → pure quant screen. Same-day caches: prices, screen, briefing, and per-name verdicts each persist in `temp/` (delete a file or tick "force refresh" to redo it). The briefing cache is keyed by its candidate sector list, so a run over a partial candidate set can never shadow the full one.

### Research audit trail

Every agent run exports its **full pass-through** to `temp/`: every step of the research process, verbatim — the prompt, every tool call with its exact search query, every tool result (including the full fetched-article text), the memo the agent wrote, and the final structured answer.

```
temp/selection_trace_briefing_{date}.md        # one per market-briefing run
temp/selection_trace_verdict_{TICKER}_{date}.md  # one per name verdict
```

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

## Reliability and security

- Price caches are range-aware, resilient to corrupt files, merged across historical cache days, and written atomically.
- Solvers validate dimensions, bounds, covariance matrices, and feasible starting weights before optimizing.
- The backtest rejects missing invested assets instead of silently renormalizing exposure, and DCA Sharpe uses cash-flow-aware returns.
- API failures receive a request ID rather than a traceback; detailed diagnostics stay in server logs. The web app adds security headers and limits concurrent expensive jobs.
- External LLM source links are restricted to HTTP(S), and the UI exposes keyboard focus, ARIA status announcements, captions, and reduced-motion behavior.

## Continuous integration

GitHub Actions runs `poetry check`, the complete pytest suite, and Playwright browser verification on pushes and pull requests. UI screenshots are uploaded as workflow artifacts for failed or reviewed runs.

For a production deployment, bind the service to localhost or place it behind authentication and a reverse proxy. The analysis endpoints run expensive synchronous work, so do not expose them directly to an untrusted network.

