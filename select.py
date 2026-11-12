"""S&P 100 selection — main entry point.

Quant funnel + LLM research (market briefing agent + per-name verdict agents).
"""

import argparse

from config.defaults import Config
from models.selection import run_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S&P 100 trending selection")
    parser.add_argument(
        "--stage",
        choices=["screen", "briefing", "verdicts", "all"],
        default="all",
        help="screen=quant only, briefing=+market agent, verdicts=+per-name agents, all=both",
    )
    parser.add_argument("--top-n", type=int, default=10, help="number of names to select (8-16)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore today's caches (prices, screen, briefing, verdicts)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config(selection_top_n=args.top_n)
    print(f"Stage           : {args.stage}")
    print(f"Nombres objetivo: {args.top_n}")
    print(f"Force refresh   : {'sí' if args.force else 'no'}")
    print()

    result = run_selection(cfg, stage=args.stage, force_refresh=args.force)
    screen = result["screen"]
    print(
        f"Fecha de corte  : {screen['as_of']}  (universe {screen['universe_size']}, "
        f"screened {screen['screened']}"
        f"{', skipped: ' + ', '.join(screen['skipped']) if screen['skipped'] else ''})"
    )

    if result["from_cache"]:
        print("(screen desde cache del día)")
    for key, message in result["degraded"].items():
        print(f"AVISO: {key} degradado - {message[:120]}")

    print(f"\n{'#' * 10} Sectores {'#' * 10}")
    for i, sector in enumerate(result["sectors"], 1):
        llm = sector.get("llm")
        llm_txt = (
            f"llm={llm['regime']} trend={llm['trend_score']:.2f} frag={llm['fragility']:.2f}"
            if llm else "llm=-"
        )
        rank = sector.get("rank", i)
        final = sector.get("final_score", sector["quant_score"])
        mark = "<<" if rank <= 3 else "  "
        print(
            f"#{rank:>2} {sector['sector']:28s} final={final:.3f} "
            f"(quant {sector['quant_score']:.3f}) {llm_txt} {mark}"
        )
        if llm:
            print(f"     driver: {llm['driver']}")

    briefing = result["briefing"]
    if briefing:
        print(f"\n{'#' * 10} Market Briefing (LLM) {'#' * 10}")
        print(f"Regimen: {briefing['regime_summary']}")
        print(f"Riesgo macro total: {briefing['overall_market_risk']:.2f}")
        for risk in briefing["macro_risks"]:
            print(f"  riesgo: {risk}")
        for theme in briefing["trending_themes"]:
            print(f"  tema: {theme}")

    if result["selection"]:
        print(f"\n{'#' * 10} Nombres seleccionados {'#' * 10}")
        for i, name in enumerate(result["selection"], 1):
            verdict = name.get("verdict")
            flag = (
                f" [ALERTA: earnings en {verdict['earnings_in_days']}d]"
                if verdict and verdict.get("event_risk") else ""
            )
            print(
                f"{i:2d}. {name['ticker']:6s} {name['name'][:24]:24s} {name['sector']:28s} "
                f"final={name['final_score']:.3f} (quant {name['finance_score']:.3f}, "
                f"llm {name['conviction'] if name['conviction'] is not None else '-'}){flag}"
            )
            if verdict:
                print(f"     tesis: {verdict['thesis']}")
                for overhang in verdict["overhangs"]:
                    print(f"     overhang: {overhang}")
        if result["vetoed"]:
            print(f"\nVetados por el LLM: {', '.join(result['vetoed'])}")
        print(
            f"\nPara optimizar:  python main.py  con shares: "
            f"{','.join(name['ticker'] for name in result['selection'])}"
        )
        print("   o el botón 'Send to Optimizer' en la página /select")


if __name__ == "__main__":
    main()
