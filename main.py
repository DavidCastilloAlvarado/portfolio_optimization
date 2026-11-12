"""Portfolio optimization pipeline — main entry point."""

from config.defaults import Config
from core.data.loader import ticker_for
from core.pipeline import run_pipeline


def main() -> None:
    cfg = Config()
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
    for name, fraction in zip(names, weights):
        symbol = ticker_for(name)
        label = f"{name} ({symbol})" if symbol else name
        print(f"{label} : {fraction * 100:.2f}% -> {round(fraction * cfg.monto_usd, 2)} USD")

    print(f"Portafolio return: {mean:.4%} -> {round(cfg.monto_usd * mean, 2)} USD")
    print(f"Portafolio standard deviation: {std:.4%} -> {round(cfg.monto_usd * std, 2)} USD")

    llm_views = result.get("llm_views")
    if llm_views:
        engine = llm_views.get("view_engine", "direct")
        print(
            f"\nLLM views: {llm_views['model']} ({engine} engine), "
            f"{llm_views['horizon_days']}-day horizon, "
            f"{llm_views['samples_per_asset']} samples/asset, "
            f"cache={'yes' if llm_views['from_cache'] else 'no'}"
        )
        if llm_views["degraded"]:
            print("LLM views unavailable - using pure equilibrium prior")
        for asset in llm_views["assets"]:
            symbol = f" ({asset['symbol']})" if asset["symbol"] else ""
            print(
                f"  {asset['ticker']}{symbol}: {asset['predicted_daily_return_pct']:+.4f}%/day "
                f"(std {asset['view_std_pct']:.4f}%, n={asset['samples_ok']})"
            )
            if asset.get("rationale"):
                print(f"    rationale: {asset['rationale']}")
                for url in asset.get("sources", [])[:3]:
                    print(f"    source: {url}")

    backtest = result.get("backtest")
    if backtest:
        print(backtest)


if __name__ == "__main__":
    main()
