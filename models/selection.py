"""S&P 100 selection: LLM layer on top of the quant funnel.

Orchestrates the three stages (screen -> briefing -> verdicts), blends the LLM
signals with the quant scores, and applies the hard gates. All LLM stages are
optional and degradable: any failure leaves the quant screen intact with a
degraded flag, mirroring the llm_bl fallback guarantees.

Blends:
  sector_final = w * quant_score + (1 - w) * trend_score * (1 - 0.5 * fragility)
  name_final   = (1 - v) * finance_score + v * conviction
  (w = Config.selection_sector_quant_weight, v = Config.selection_name_llm_weight)

Hard gates (no score can override):
  - a name's sector must be among the top 3 blended sectors
  - an LLM verdict with include=false vetoes the name
  - earnings inside Config.selection_event_risk_days flag the name (kept, flagged)
"""

from config.defaults import Config
from config.llm import LLMSettings, WebSearchMCPSettings
from core import selection as core_selection
from core.progress import progress_log
from models.selection_agents import (
    MarketBriefing,
    SelectionVerdict,
    run_market_briefing,
    run_name_verdicts,
)

TOP_SECTORS = 3
BRIEFING_SECTOR_CANDIDATES = 8


def _blend_sectors(screen: dict, briefing: MarketBriefing | None, cfg: Config) -> tuple[list[dict], bool]:
    llm_by_sector = {s.sector: s for s in (briefing.sectors if briefing else [])}
    out = []
    for s in screen["sectors"]:
        sector_llm = llm_by_sector.get(s["sector"])
        if sector_llm is not None:
            llm_component = sector_llm.trend_score * (1 - 0.5 * sector_llm.fragility)
            final = cfg.selection_sector_quant_weight * s["quant_score"] + (1 - cfg.selection_sector_quant_weight) * llm_component
            llm = {
                "trend_score": round(sector_llm.trend_score, 4),
                "regime": sector_llm.regime,
                "driver": sector_llm.driver,
                "fragility": round(sector_llm.fragility, 4),
                "conviction": round(sector_llm.conviction, 4),
            }
        else:
            final = s["quant_score"]
            llm = None
        out.append({**s, "llm": llm, "final_score": round(final, 4)})
    out.sort(key=lambda s: (-s["final_score"], s["sector"]))
    for i, s in enumerate(out):
        s["rank"] = i + 1
    expected = {s["sector"] for s in screen["sectors"][:BRIEFING_SECTOR_CANDIDATES]}
    degraded = briefing is None or not expected.issubset(llm_by_sector)
    return out, degraded


def _event_flag(name_entry: dict, verdict: SelectionVerdict | None, cfg: Config) -> dict:
    days = verdict.earnings_in_days if verdict is not None else None
    return {
        "earnings_in_days": days,
        "event_risk": days is not None and days <= cfg.selection_event_risk_days,
    }


def _select_names(
    screen: dict,
    sectors_final: list[dict],
    verdicts: dict[str, SelectionVerdict],
    cfg: Config,
) -> tuple[list[dict], set[str]]:
    hot_sectors = {s["sector"] for s in sectors_final[:TOP_SECTORS]}
    vetoed: set[str] = set()
    selected = []
    for entry in screen["names"]:
        if entry["deduped"] or entry["sector"] not in hot_sectors:
            continue
        verdict = verdicts.get(entry["ticker"])
        if verdict is not None and not verdict.include:
            vetoed.add(entry["ticker"])
            continue
        final = (1 - cfg.selection_name_llm_weight) * entry["finance_score"]
        if verdict is not None:
            final += cfg.selection_name_llm_weight * verdict.conviction
        selected.append({
            **entry,
            "conviction": round(verdict.conviction, 4) if verdict is not None else None,
            "final_score": round(final, 4),
            "verdict": None if verdict is None else {
                "include": verdict.include,
                "thesis": verdict.thesis,
                "catalysts": list(verdict.catalysts),
                "overhangs": list(verdict.overhangs),
                **_event_flag(entry, verdict, cfg),
            },
        })
    selected.sort(key=lambda n: (-n["final_score"], -n["finance_score"], n["ticker"]))
    return selected[:cfg.selection_top_n], vetoed


def run_selection(
    cfg: Config,
    stage: str = "all",
    settings: LLMSettings | None = None,
    mcp_settings: WebSearchMCPSettings | None = None,
    force_refresh: bool = False,
) -> dict:
    """Run the selection funnel up to the requested stage.

    Stages: "screen" (quant only), "briefing" (screen + market agent),
    "verdicts" (screen + per-name agents), "all" (briefing + verdicts).

    Returns:
        {stage, as_of, screen, briefing, sectors, selection, vetoed,
         from_cache, degraded}
    """
    if stage not in ("screen", "briefing", "verdicts", "all"):
        raise ValueError(f"Unknown selection stage: {stage}")

    progress_log("select", f"funnel start | stage={stage} top_n={cfg.selection_top_n}")
    screen = core_selection.run_screen(cfg, force_refresh=force_refresh)
    progress_log(
        "select",
        f"screen: {len(screen['names'])} names x {len(screen['sectors'])} sectors (from_cache={screen['from_cache']})",
    )
    result = {
        "stage": stage,
        "as_of": screen["as_of"],
        "screen": screen,
        "briefing": None,
        "sectors": list(screen["sectors"]),
        "selection": [],
        "vetoed": [],
        "from_cache": screen["from_cache"],
        "degraded": {},
    }
    if stage == "screen":
        return result

    briefing: MarketBriefing | None = None
    if stage in ("briefing", "all"):
        candidates = [s["sector"] for s in screen["sectors"][:BRIEFING_SECTOR_CANDIDATES]]
        progress_log("select", f"phase: market briefing ({len(candidates)} candidate sectors)")
        try:
            briefing = run_market_briefing(candidates, cfg, settings=settings, mcp_settings=mcp_settings, force_refresh=force_refresh)
            result["briefing"] = {
                "regime_summary": briefing.regime_summary,
                "macro_risks": list(briefing.macro_risks),
                "trending_themes": list(briefing.trending_themes),
                "overall_market_risk": briefing.overall_market_risk,
            }
        except Exception as exc:
            result["degraded"]["briefing"] = str(exc)
            progress_log("select", f"briefing FAILED ({exc}) — sectors stay quant-only")

    result["sectors"], sectors_degraded = _blend_sectors(screen, briefing, cfg)
    if sectors_degraded:
        result["degraded"]["sectors"] = "briefing unavailable - quant scores only"
    top3 = ", ".join(f"{s['sector']} {s['final_score']:.3f}" for s in result["sectors"][:TOP_SECTORS])
    progress_log("select", f"blended sectors | top {TOP_SECTORS}: {top3}")

    verdicts: dict[str, SelectionVerdict] = {}
    if stage in ("verdicts", "all"):
        hot_sectors = {s["sector"] for s in result["sectors"][:TOP_SECTORS]}
        candidates = [e for e in screen["names"] if not e["deduped"] and e["sector"] in hot_sectors][:cfg.selection_top_n * 2]
        progress_log("select", f"phase: name verdicts ({len(candidates)} candidates)")
        try:
            verdicts = run_name_verdicts(candidates, cfg, settings=settings, mcp_settings=mcp_settings, force_refresh=force_refresh)
        except Exception as exc:
            result["degraded"]["verdicts"] = str(exc)
            progress_log("select", f"verdicts FAILED ({exc}) — names stay quant-only")

    selected, vetoed = _select_names(screen, result["sectors"], verdicts, cfg)
    result["selection"] = selected
    result["vetoed"] = sorted(vetoed)
    progress_log(
        "select",
        f"funnel done | {len(selected)} selected | vetoed {len(vetoed)} | "
        f"{', '.join(n['ticker'] for n in selected) or '-'}",
    )
    return result
