# Portfolio Optimization Engine

A Python tool that finds the optimal asset allocation for your investment portfolio using modern portfolio theory — balancing **risk** and **return** through simulation and optimization.

## What it does

1. **Fetches historical price data** from Yahoo Finance for any set of stock tickers or ETFs
2. **Optimizes weights** using either Mean-Variance (max Sharpe ratio) or Minimal Variance strategies
3. **Simulates outcomes** via Monte Carlo and backtests against real historical data

---

## Quickstart

### Install dependencies

```bash
poetry install
```

The project requires Python 3.10+ and the following libraries:

| Library | Purpose | Reference |
|---------|---------|-----------|
| [pandas](https://pandas.pydata.org/) | Data manipulation and time-series handling | https://pandas.pydata.org/ |
| [numpy](https://numpy.org/) | Numerical computing (matrix ops, statistics) | https://numpy.org/ |
| [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) | SLSQP solver for Mean-Variance optimization | https://docs.scipy.org/doc/scipy/ |
| [cvxopt](https://cvxopt.org/) | Quadratic programming for Minimal Variance | https://cvxopt.org/ |
| [requests](https://requests.readthedocs.io/) | HTTP client to fetch data from Yahoo Finance API | https://requests.readthedocs.io/ |
| [tqdm](https://tqdm.github.io/) | Progress bars for bulk downloads | https://tqdm.github.io/ |

### Run the script

```bash
poetry run python main.py
```

---

## How `main.py` works

The script is organized into four logical sections:

### 1. Configuration (lines 9–28)

Defines all tunable parameters at the top of the file so you can customize without touching the logic below:

| Variable | Description |
|----------|-------------|
| `WEEK` / `MONTH` | Resample frequency (`None`, `"week"`, or `"month"`). `None` = daily. |
| `MIN_VARIANCE` | `True` = Minimal Variance strategy; `False` = Mean-Variance (max Sharpe) |
| `MONTOUSD` | Initial investment amount in USD |
| `MONTHLY_DELTA` | Monthly dollar-amount to add each period (DCA contribution) |
| `SHARES` | List of ticker symbols (stocks or ETFs) to include in the portfolio |
| `W_LIMITS` | Min and max weight constraint per asset (default: 2%–12%) |
| `DAYS` | Calendar days of historical data to analyze for computing returns/covariance |
| `SIM_DAYS` | Trading days to simulate (~252 = ~1 year) |
| `RISK_FREE_ANUL_PERC` | Annual risk-free rate in percent (default: 5%) |

### 2. Data Loading (lines 33–43)

Uses functions from `data_loader.py`:

- **`bulk_stocks(SHARES, DAYS)`** — Fetches daily close prices for all tickers via the [Yahoo Finance Query v8 API](https://github.com/LearnCash/finance-yahoo). Results are cached locally under `temp/` to avoid repeated network calls.
- **`prepare_returns(raw_data, resample)`** — Sorts data by date, interpolates missing values using time-weighted interpolation, and computes percentage returns (daily, weekly, or monthly depending on your config).

From the processed data, the script derives:
- `mean_returns` — expected daily return per asset
- `cov_returns` — covariance matrix of daily returns across assets

### 3. Optimization (lines 47–61)

Calls `optimize()` from `optimizer.py`, which supports two strategies:

- **Mean-Variance (Sharpe)** — Uses `scipy.optimize.minimize` with the SLSQP method to maximize the Sharpe ratio by minimizing its inverse. Based on [Harry Markowitz's Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory), first published in 1952.
- **Minimal Variance** — Uses `cvxopt.qp` (quadratic programming) to find the portfolio with the lowest possible variance subject to weight constraints.

The Sharpe ratio, defined as `(portfolio_return - risk_free_rate) / portfolio_std`, was introduced by [William F. Sharpe](https://en.wikipedia.org/wiki/William_F._Sharpe) in 1966 and remains one of the most widely used risk-adjusted performance measures.

Output includes:
- Each asset's optimal weight percentage and allocated USD amount
- Portfolio expected return (mean) and volatility (standard deviation)

### 4. Simulations (lines 66–72)

Runs two simulation modes via `simulation.py`:

#### Monte Carlo Simulation (`print_rendimiento`)
Generates thousands of random future scenarios using a normal distribution parameterized by the portfolio's mean return and standard deviation. Outputs descriptive statistics of simulated returns across all runs.

Based on the [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method), widely used in quantitative finance for stochastic modeling. See also: [Hull, John C. — Options, Futures, and Other Derivatives](https://www.prenticehall.com/business/engineering/Options_Futures_and_Other_Derivatives/product/fullpage/0136089014).

#### Historical Backtest (`print_rendimiento_backtest`)
Uses real cached price data to simulate portfolio performance under two investment approaches:

- **Lump-sum** — Invests the full `MONTOUSD` upfront
- **DCA (Dollar Cost Averaging)** — Adds `MONTHLY_DELTA` every ~21 trading days

Reports per mode:
- Final portfolio value and total return percentage
- Maximum drawdown
- Annualized Sharpe ratio

---

## Key Concepts & References

| Concept | Reference |
|---------|-----------|
| Modern Portfolio Theory (Markowitz, 1952) | [Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory) · [Original Paper](https://www.aeaweb.org/articles?id=10.1257/aer.40.3.77) |
| Sharpe Ratio (Sharpe, 1966) | [Wikipedia](https://en.wikipedia.org/wiki/Sharpe_ratio) · [Original Paper](https://www.ms.jhu.edu/~newham/math171/sharpe.pdf) |
| Minimal Variance Portfolio | [Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory#Minimum_variance_portfolio) |
| Monte Carlo Simulation in Finance | [Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method_in_finance) |
| Dollar-Cost Averaging | [Wikipedia](https://en.wikipedia.org/wiki/Dollar-cost_averaging) · [Investopedia](https://www.investopedia.com/terms/d/dollarcostaveraging.asp) |
| Yahoo Finance API v8 | [GitHub Mirror](https://github.com/LearnCash/finance-yahoo) |

---

## Customizing Your Portfolio

Edit the `SHARES` list in `main.py` to include any tickers supported by Yahoo Finance. Examples:

```python
# US equities and ETFs
SHARES = ['SPY', 'QQQ', 'VTI', 'BND']

# International exposure
SHARES = ['VEA', 'VWO', 'EWJ', 'EEM']
```

Adjust `W_LIMITS` to control concentration risk, and set `MIN_VARIANCE = True` if you prefer a conservative low-volatility allocation over maximizing the Sharpe ratio.
