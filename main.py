"""Portfolio optimization pipeline — main entry point."""

import numpy as np
from config.defaults import Config
from data_loader import bulk_stocks, prepare_returns
from optimizer import optimize
from simulation import run_backtest

# ── Default configuration ─────────────────────────────────────────────
cfg = Config()

# ── Data loading ──────────────────────────────────────────────────────
resample = cfg.get_resample()
raw_data = bulk_stocks(cfg.shares, cfg.days)
data, returns = prepare_returns(raw_data, resample=resample)

names = data.columns.tolist()
records = len(data)

mean_returns = np.array(returns.mean())
cov_returns = np.array(returns.cov())

# ── Optimization ──────────────────────────────────────────────────────
print(f"Días de análisis : {cfg.days}")
print(f"Cantidad de records analizados: {records}")
print(f"Monto total de inversión: {cfg.monto_usd} usd")

weights, mean, std, strategy = optimize(
    mean_returns.copy(), cov_returns.copy(), cfg.risk_free, cfg.w_limits, cfg.min_variance,
)

print(f"\n{'#' * 10} {strategy} {'#' * 10}")
for name, fp in zip(names, weights):
    print(f"{name} : {fp * 100:.2f}% -> {round(fp * cfg.monto_usd, 2)} USD")

print(f"Portafolio return: {mean:.4%} -> {round(cfg.monto_usd * mean, 2)} USD")
print(f"Portafolio standard deviation: {std:.4%} -> {round(cfg.monto_usd * std, 2)} USD")

# ── Backtest ──────────────────────────────────────────────────────────
backtest = run_backtest(
    cfg.monto_usd, cfg.shares, weights, cfg.sim_days,
    cfg.risk_free_annual_perc / 100, cfg.monthly_delta,
)
print(backtest)
