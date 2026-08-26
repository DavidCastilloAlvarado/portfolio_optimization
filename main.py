"""Portfolio optimization pipeline — main entry point."""

from config.defaults import Config
from core.data.loader import ticker_for
from core.pipeline import run_pipeline

# ── Default configuration ─────────────────────────────────────────────
cfg = Config()

# ── Pipeline ──────────────────────────────────────────────────────────
print(f"Días de análisis : {cfg.days}")
print(f"Monto total de inversión: {cfg.monto_usd} usd")

result = run_pipeline(cfg)

names = result["names"]
weights = result["weights"]
mean = result["mean"]
std = result["std"]
strategy = result["strategy"]

print(f"Cantidad de records analizados: {result['records']}")

print(f"\n{'#' * 10} {strategy} {'#' * 10}")
for name, fp in zip(names, weights):
    symbol = ticker_for(name)
    label = f"{name} ({symbol})" if symbol else name
    print(f"{label} : {fp * 100:.2f}% -> {round(fp * cfg.monto_usd, 2)} USD")

print(f"Portafolio return: {mean:.4%} -> {round(cfg.monto_usd * mean, 2)} USD")
print(f"Portafolio standard deviation: {std:.4%} -> {round(cfg.monto_usd * std, 2)} USD")

# ── Backtest ──────────────────────────────────────────────────────────
backtest = result.get("backtest")
if backtest:
    print(backtest)
