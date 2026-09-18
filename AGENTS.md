# AGENTS.md — Portfolio Optimization Engine

## Project Overview

A Python portfolio optimization engine using Modern Portfolio Theory with four strategies: Mean-Variance (Sharpe), Minimal Variance, Growth (Kelly), and LLM-Black-Litterman (LLM return forecasts blended with a market-equilibrium prior via Bayesian updating). Includes a FastAPI web UI, multi-source market data (Yahoo Finance primary, JustETF fallback for ISINs) with CSV caching, and historical backtesting (lump-sum + DCA).

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
│   ├── defaults.py         # Config dataclass (all tunable parameters)
│   ├── llm.py              # LLMSettings — LLM endpoint from env vars (LLM_BASE_URL, LLM_MODEL, ...)
│   └── isin_ticker_map.json  # ISIN → display ticker map (user-editable JSON, no code)
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
│   ├── blm.py              # LLM-Black-Litterman: prior, posterior, LLM view generation + daily cache
│   ├── llm_client.py       # OpenAI-compatible chat-completions HTTP client (requests)
│   └── blitterman.py       # Legacy Black-Litterman + market cap utilities (not used by the llm_bl strategy)
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
    ├── test_web_api.py     # FastAPI endpoints (mocked data)
    ├── test_llm_client.py  # OpenAI-compatible client (mocked HTTP)
    └── test_blm.py         # Black-Litterman math, view parsing, cache, thinking handling (mocked LLM)
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
- **ISIN → display ticker**: the map lives in `config/isin_ticker_map.json` (plain JSON, `"ISIN": "TICKER"` pairs — users add entries there without touching code). `core/data/loader.py` loads it at import into `ISIN_TICKER_MAP` via `_load_isin_ticker_map()` (missing file or invalid JSON raise with a message pointing at the file); `ticker_for()` resolves a display ticker. Single source of truth: the CLI (`main.py`), the web renderer (`symbol` field per weight) and the browser UI (injected as `__ISIN_MAP__` → `window.ISIN_TICKER_MAP`) all resolve from it. Data fetching still keys on the ISIN — the map is display-only.

### Optimization
- `core/optimization/solvers.py` exports `optimize()` as the main entry point
- Four strategies (selected by `Config.strategy`, resolved by `Config.resolve_strategy()`):
  - `sharpe` — Mean-Variance (Sharpe), default
  - `min_variance` — Minimal Variance
  - `kelly` — Growth (Kelly): maximizes `w'μ − (1/2λ)w'Σw`, solved via `cvxopt` QP; `kelly_fraction` scales risk (1.0 = full Kelly, 0.5 = half-Kelly default)
  - `llm_bl` — LLM-Black-Litterman (see dedicated section below; the posterior mean is fed to the Sharpe MVO solver)
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

## LLM-Black-Litterman Strategy (`llm_bl`)

Blends LLM return forecasts with a market-equilibrium prior using Bayesian
(Black-Litterman) updating. The posterior mean is handed to the existing Sharpe
MVO solver. This is an additive technique (not a replacement) and **always runs
on daily-frequency returns** (recomputed from `raw_data` via
`prepare_returns(raw_data, resample=None)`) so view and covariance units stay
consistent, regardless of `cfg.resample`.

### Files
- `models/llm_client.py` — `chat_completion()`: OpenAI-compatible
  `/chat/completions` HTTP call (`requests`); bearer header only when
  `LLM_API_KEY` is set; `RuntimeError` on HTTP/timeout/malformed.
- `config/llm.py` — `LLMSettings.from_env()`.
- `models/blm.py` — prior, posterior, view generation + caching.

### Math
- **Equilibrium prior**: `π = τ·(Σ ⊙ w)` where `w` is equal or inverse-volatility
  (`blm_prior`), scaled by `τ` (`llm_tau`). **No market-cap data** is used.
- **Views**: for each asset, the LLM is sampled `llm_samples` times; view
  `q` = sample mean, `Ω_ii` = sample variance (of the *forecast distribution*,
  not the asset's own history).
- **Posterior**: `μ_BL = [ (τΣ)⁻¹ + P'Ω⁻¹P ]⁻¹ [ (τΣ)⁻¹π + P'Ω⁻¹q ]`,
  solved with a `pinv` fallback (1e-10 regularization) for ill-conditioning.

### Robustness / design decisions (improvements over the reference repo)
- **JSON-first parsing** with a numeric-regex fallback (`_parse_view_response`).
- **Ω floor**: `Ω_ii = max(sample_var, OMEGA_FLOOR_FRACTION·τ·Σ_ii)`
  (`OMEGA_FLOOR_FRACTION = 0.25`) — guards against over-confident (near-zero
  variance) views and singular matrices.
- **Partial views**: assets with no valid samples are excluded from `P` (viewed
  set is a subset); the posterior is still well-defined.
- **Graceful degradation**: if the LLM is unreachable or every view is garbage,
  the run falls back to the **pure equilibrium prior** (`degraded=True`) and the
  pipeline still returns weights.
- **Daily view cache**: `llm_views_{sha1-12}_{YYYY-MM-DD}.json` in `temp/`, keyed
  by model|samples|tau|prior|horizon|history|names. Same-day re-runs reuse
  views (zero extra API cost). Tests monkeypatch `core.data.cache.CACHE_DIR`.
- **Fixed τ** (no validation grid search) — honest for live use.
- **Per-asset sampling is sequential** (gentle on the local 27B model).
- **Sample plausibility clamp**: a sample with |value| > `MAX_PLAUSIBLE_DAILY_PCT`
  (50 %/day) is rejected as implausible.

### Reasoning-model (thinking) handling
vLLM-served reasoning models (e.g. Qwen3) spend their whole token budget in
`reasoning_content` and return empty `content`, so views would all fail.
`models/blm.py::_call_for_sample` therefore sends
`chat_template_kwargs={"enable_thinking": false}` for fast direct answers
(`VIEW_MAX_TOKENS_FAST=512`) and retries once without it if the endpoint rejects
the parameter (remembered in-process). Set `LLM_ENABLE_THINKING=1` to force
thinking back on (uses `VIEW_MAX_TOKENS_THINKING=2048`).

### LLM environment variables (`config/llm.py`)
- `LLM_BASE_URL` (default `http://192.168.18.200:1234/v1`)
- `LLM_API_KEY` (default empty — local servers need none)
- `LLM_MODEL` (default `Qwen3.8_mtp-27B`)
- `LLM_TIMEOUT` (default `120`)
- `LLM_ENABLE_THINKING` (optional, `1` to re-enable model reasoning)

### Web / CLI
- Web form fields: `llm_samples`, `llm_tau`, `blm_prior`, `blm_horizon_days`
  (panel shown only when `strategy=llm_bl`). The response's `optimization` dict
  gains an `llm_views` key (`assets[]` with predicted daily return, view std,
  sample count) rendered as an "LLM Views" table in the UI.
- CLI (`main.py`) prints an "LLM views" section when present.
- The strategy label is `"Black-Litterman Optimization (LLM views)"`, or
  `"(equilibrium fallback)"` when degraded.
