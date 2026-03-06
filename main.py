"""Portfolio optimization pipeline — main entry point."""

# %%
import numpy as np
from data_loader import bulk_stocks, prepare_returns
from optimizer import optimize
from simulation import print_rendimiento, print_rendimiento_backtest

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
WEEK = False
MONTH = False
MIN_VARIANCE = False  # True = Minimal Variance, False = Mean-Variance (Sharpe)

MONTOUSD = 2300
MONTHLY_DELTA = 500   # USD added monthly (DCA)

SHARES = [
     'XLU', 'QQQ', 'SCHD',  'GLD',  'JEPQ', 'XLE',
    'AAPL', 'MSFT', 'AMD','GOOG',
]
W_LIMITS = (0.00, 0.18)

DAYS = 720            # calendar days for data analysis
SIM_DAYS = 252        # trading days for simulation (~1 year)
RISK_FREE_ANUL_PERC = 5
RISK_FREE = (1 + RISK_FREE_ANUL_PERC / 100) ** (1 / 365) - 1

# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════
# %%
resample = "week" if WEEK else ("month" if MONTH else None)
raw_data = bulk_stocks(SHARES, DAYS)
data, returns = prepare_returns(raw_data, resample=resample)

names = data.columns.tolist()
RECORDS = len(data)

mean_returns = np.array(returns.mean())
cov_returns = np.array(returns.cov())

# ══════════════════════════════════════════════════════════════
#  OPTIMIZATION
# ══════════════════════════════════════════════════════════════
# %%
print(f"Días de análisis : {DAYS}")
print(f"Cantidad de records analizados: {RECORDS}")
print(f"Monto total de inversión: {MONTOUSD} usd")

weights, mean, std, strategy = optimize(
    mean_returns.copy(), cov_returns.copy(), RISK_FREE, W_LIMITS, MIN_VARIANCE,
)

print(f"\n{'#'*10} {strategy} {'#'*10}")
for name, fp in zip(names, weights):
    print(f"{name} : {fp*100:.2f}% -> {round(fp*MONTOUSD, 2)} USD")

print(f"Portafolio return: {mean:.4%} -> {round(MONTOUSD*mean, 2)} USD")
print(f"Portafolio standard deviation: {std:.4%} -> {round(MONTOUSD*std, 2)} USD")

# ══════════════════════════════════════════════════════════════
#  SIMULATIONS
# ══════════════════════════════════════════════════════════════
# %%
print_rendimiento(MONTOUSD, SIM_DAYS, mean, std, 4000)
print_rendimiento_backtest(
    MONTOUSD, SHARES, weights, SIM_DAYS, RISK_FREE_ANUL_PERC / 100, MONTHLY_DELTA,
)
print("#" * 50)
# %%
