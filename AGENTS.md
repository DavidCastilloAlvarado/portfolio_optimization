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
├── .env                    # Local LLM/MCP values (gitignored; template: .env.example)
├── .env.example            # Template for .env (LLM endpoint + web-search MCP)
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
│   ├── blm.py              # LLM-Black-Litterman: prior, posterior, view engines dispatch + daily cache
│   ├── llm_agent.py        # Research-agent view engine: LangGraph agent + web-search MCP + JSON contract
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
- `optimize()` raises `ValueError` up front when the bounds leave an empty
  feasible set (e.g. 2 assets × 12% max weight sums to 0.24 < 1.0) — the web
  UI surfaces the message; raise the max weight or add tickers

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
- `config/llm.py` — `LLMSettings.from_env()`, `WebSearchMCPSettings.from_env()`.
- `models/blm.py` — prior, posterior, view-engine dispatch + daily cache.
- `models/llm_agent.py` — research-agent view engine (LangGraph agent + MCP
  web search + strict `ViewContract` JSON contract).

### Math
- **Equilibrium prior**: `π = τ·(Σ ⊙ w)` where `w` is equal or inverse-volatility
  (`blm_prior`), scaled by `τ` (`llm_tau`). **No market-cap data** is used.
- **Views**: for each asset, the LLM is sampled `llm_samples` times; view
  `q` = sample mean, `Ω_ii` = sample variance (of the *forecast distribution*,
  not the asset's own history).
- **Posterior**: `μ_BL = [ (τΣ)⁻¹ + P'Ω⁻¹P ]⁻¹ [ (τΣ)⁻¹π + P'Ω⁻¹q ]`,
  solved with a `pinv` fallback (1e-10 regularization) for ill-conditioning.

### View engines (`llm_view_engine`)
Two engines produce the same `(q, Ω)` sample statistics; `run_blm` dispatches
via `_generate_engine_views` (agent is lazy-imported from `models/blm.py` to
avoid a circular import):

- **`agent` (default)** — a LangGraph ReAct agent
  (`langchain.agents.create_agent(llm, tools, system_prompt=..., response_format=ViewContract)`)
  that MUST web-research the ticker (latest news, analyst consensus, latest
  earnings/guidance, macro/sector) before forecasting, and returns the forecast
  as the strict Pydantic `ViewContract`
  (`predicted_avg_daily_return_pct`, `confidence` 0-1, `rationale`, `sources`).
  The agent's tool list is the MCP search tools **plus two local research-cache
  tools** (`load_research`, `save_research`): the protocol is cache-first —
  call `load_research(ticker)` first; if today's markdown notes exist, reuse
  them and do NOT re-search, otherwise research and then
  `save_research(ticker, notes)`. The notes are required to be a **long,
  detailed, self-contained analyst memo** (company snapshot, dated news,
  per-firm analyst targets, verbatim earnings/guidance figures, catalysts,
  macro/sector, bull/bear case, and a numbered `Sources` list of ALL urls
  used, cited inline) so any later forecast is reconstructable from the file
  alone. Notes persist in
  `temp/research_{TICKER}_{YYYY-MM-DD}.md`, so research is paid **once per
  ticker per day** and later samples/runs (even with different sample counts)
  only re-reason from the cached evidence. Delete the file to force fresh
  research. The model runs with reasoning enabled via
  `extra_body={"reasoning_effort": cfg.llm_reasoning_effort}` (default
  `"medium"`; wire values `none|low|medium|high|xhigh` — verified effective
  on the LM Studio llama.cpp endpoint), and the per-LLM-call completion budget
  is `llm_agent_max_tokens` (default 32768, clamped 512–32768) sized to fit
  the full research note — note the budget covers **reasoning + output** of
  each LLM call (verified: a 50-token cap with thinking on returns empty
  `content`, all tokens in `reasoning_content`); the local model handles
  200k context / 30k output.
  Tools come from the chinchilla web-search
  MCP server over stdio via `langchain.mcp.MCPAdapter` (LangChain 1.x
  `MCPConfig` target — the `langchain-mcp-adapters` package is archived):
  `npx -y dsh-google-chrome-search mcp` with `CHROME_PATH` + `GSEARCH_PROFILE`
  env (same server/profile the developer workstation uses; Google via local
  Chrome CDP, Readability extraction; CAPTCHAs open a visible window).
  One MCP session is shared across all assets/samples of a run. Per-agent-call
  timeout `AGENT_SAMPLE_TIMEOUT_S` (1200 s); the per-sample tool-call budget is
  `Config.llm_agent_recursion_limit` (default 40, clamped 5–200) LangGraph
  supersteps — N tool calls consume 2N+1 steps, so 40 allows 19 tool calls.
  Each failed
  sample is retried once (transient local-LLM slowness is common) before the
  per-asset fallback kicks in. Timing: the first sample of a day generates the
  exhaustive note (~10-20 min), later samples only re-reason from it
  (~2-4 min), so `llm_samples` is the main cost knob and same-day re-runs are
  cheap.
- **`direct`** — legacy fast path: history-only prompt, thinking disabled
  (see below), no web access.

Agent fallbacks (all preserve the existing guarantees): per-asset agent
failure (timeout, MCP down, malformed contract) → that asset is sampled via
the direct engine; agent engine unavailable (MCP command unset /
`llm_agent_websearch` off / infrastructure error) → the whole run uses the
direct engine; everything failing → pure equilibrium prior (`degraded=True`).
Agent-sourced views additionally carry per-sample `details` (rationale,
sources, confidence) — stored in the daily cache and surfaced in UI/CLI.

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
  by model|engine|effort|websearch|samples|tau|prior|horizon|history|names
  (engine-aware so agent and direct views never cross-reuse). Same-day re-runs
  reuse views (zero extra API cost). Tests monkeypatch `core.data.cache.CACHE_DIR`.
- **Fixed τ** (no validation grid search) — honest for live use.
- **Per-asset sampling is sequential** (gentle on the local 27B model).
- **Sample plausibility clamp**: a sample with |value| > `MAX_PLAUSIBLE_DAILY_PCT`
  (50 %/day) is rejected as implausible.

### Reasoning-model (thinking) handling
The direct engine needs fast, parseable answers: vLLM/llama.cpp-served
reasoning models (e.g. Qwen3) would otherwise spend their whole token budget in
`reasoning_content` and return empty `content`. `models/blm.py::_call_for_sample`
therefore sends `chat_template_kwargs={"enable_thinking": false}`
(`VIEW_MAX_TOKENS_FAST=512`) and retries once without it if the endpoint rejects
the parameter (remembered in-process). Set `LLM_ENABLE_THINKING=1` to force
thinking back on (uses `VIEW_MAX_TOKENS_THINKING=2048`).

The agent engine does the opposite on purpose: reasoning is **enabled** via
`reasoning_effort` (`Config.llm_reasoning_effort`, default `"medium"`) because
multi-step research benefits from it — the model reasons over its gathered web
evidence before emitting the contract. Empirically verified on the local
llama.cpp server: `low` ≈ 400 chars of reasoning, `high` ≈ 1100.

### LLM settings (project `.env`)
Machine-specific LLM/MCP values live in the project-root **`.env`** file
(gitignored; committed template is **`.env.example`**), loaded via
python-dotenv when `config/llm.py` is imported. Real environment variables
always take precedence over `.env`. No machine-specific values exist in code:

- `LLM_BASE_URL` (e.g. `http://192.168.18.200:1234/v1`) — required for `llm_bl`;
  unset → `is_configured == False` → degrades to the pure equilibrium prior
- `LLM_API_KEY` (empty — local servers need none)
- `LLM_MODEL` (e.g. `Qwen3.8_mtp-27B`) — required for `llm_bl`
- `LLM_TIMEOUT` (default `120`)
- `LLM_ENABLE_THINKING` (optional, `1` to re-enable model reasoning in direct mode)
- `WEBSEARCH_MCP_COMMAND` (e.g. `/home/david/.local/node/bin/npx`) — stdio
  command launching the web-search MCP server for the agent engine;
  unset → agent engine unavailable → direct engine
- `WEBSEARCH_MCP_ARGS` (default `-y dsh-google-chrome-search mcp`)
- `CHROME_PATH` (default `/usr/bin/google-chrome`) and `GSEARCH_PROFILE`
  (default `~/.dsh-chrome-google`) — passed to the MCP server

### Web / CLI
- Web form fields: `llm_samples`, `llm_tau`, `blm_prior`, `blm_horizon_days`,
  `llm_view_engine` (agent/direct), `llm_reasoning_effort`
  (none/low/medium/high/xhigh), `llm_agent_websearch`, `llm_agent_max_tokens`,
  `llm_agent_recursion_limit` (panel shown only when `strategy=llm_bl`;
  max_tokens/recursion_limit are power knobs without UI inputs, like websearch). The `/optimize` endpoint is a
  **sync** `def` on purpose: FastAPI runs it in the threadpool so the agent
  engine can open its own asyncio event loop for the MCP session. The
  response's `optimization` dict gains an `llm_views` key (`view_engine`,
  `assets[]` with predicted daily return, view std, sample count, and for
  agent views `rationale` + `sources` + `confidence`) rendered as an "LLM
  Views" table in the UI.
- CLI (`main.py`) prints an "LLM views" section (per-asset rationale and
  top sources when present).
- The strategy label is `"Black-Litterman Optimization (LLM views)"`, or
  `"(equilibrium fallback)"` when degraded.
