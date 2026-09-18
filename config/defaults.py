"""Portfolio optimization configuration."""

from dataclasses import dataclass, field
from typing import List


def _parse_limit_side(s: str):
    """Parse one side of a per-ticker bound; empty/invalid -> None (use global)."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int(raw, default: int, lo: int, hi: int) -> int:
    """Parse an int from form data; empty/invalid -> default, clamped to [lo, hi]."""
    try:
        val = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return min(max(val, lo), hi)


def _parse_float(raw, default: float, lo: float, hi: float) -> float:
    """Parse a float from form data; empty/invalid -> default, clamped to [lo, hi]."""
    try:
        val = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return min(max(val, lo), hi)


@dataclass
class Config:
    """All tunable parameters for the portfolio optimization pipeline."""

    # ── Data ──────────────────────────────────────────────────────────
    resample: str = "none"  # "none", "week", "month"
    days: int = 720  # calendar days for historical data

    # ── Assets ────────────────────────────────────────────────────────
    shares: List[str] = field(default_factory=lambda: [
        "XLU", "QQQ", "SCHD", "GLDM", "SPY",
        "AAPL", "TSM", "AMD", "GOOG",
        "IE00BFMXXD54", "IE00B53SZB19", "IE00B4ND3602",
    ])
    w_limits: tuple = (0.02, 0.12)  # (min_weight, max_weight)
    w_limits_per_ticker: dict = field(default_factory=dict)  # ticker -> (min, max); None side falls back to global

    # ── Optimization ─────────────────────────────────────────────────
    min_variance: bool = False  # Legacy flag; strategy takes precedence when set (kept for backward compatibility)
    strategy: str = "sharpe"  # "sharpe" | "min_variance" | "kelly" | "llm_bl"
    kelly_fraction: float = 0.5  # Kelly aggressiveness λ: 1.0 = full Kelly, 0.5 = half-Kelly (recommended)
    llm_samples: int = 5  # LLM samples per asset for Black-Litterman view confidence
    llm_tau: float = 1.0 / 26.0  # Black-Litterman τ: scales prior covariance (1/26 ≈ biweekly rebalance)
    blm_prior: str = "equal"  # "equal" | "inverse_vol": equilibrium prior weights
    blm_horizon_days: int = 14  # Forecast horizon in trading days for LLM views
    blm_history_days: int = 56  # History window (trading days) given to the LLM
    llm_view_engine: str = "agent"  # "agent" (LangGraph + web research) | "direct" (history-only samples)
    llm_reasoning_effort: str = "medium"  # "none" | "low" | "medium" | "high" | "xhigh": model reasoning effort
    llm_agent_websearch: bool = True  # Agent engine: require the web-search MCP server (else fall back to direct)
    llm_agent_max_tokens: int = 32768  # Completion token budget per agent LLM call, reasoning + output (fits long research notes)
    llm_agent_recursion_limit: int = 40  # LangGraph supersteps per agent sample (2N+1 steps for N tool calls)

    # ── Investment ────────────────────────────────────────────────────
    monto_usd: float = 10000.0
    monthly_delta: float = 300.0

    # ── Simulation ────────────────────────────────────────────────────
    sim_days: int = 252  # trading days for simulation (~1 year)
    risk_free_annual_perc: float = 5.0

    # ── Computed ──────────────────────────────────────────────────────
    @property
    def risk_free(self) -> float:
        """Daily risk-free rate derived from annual percentage on a 252-trading-day basis."""
        return (1 + self.risk_free_annual_perc / 100) ** (1 / 252) - 1

    def get_resample(self) -> str | None:
        """Return resample string; 'none' → None."""
        if self.resample == "none":
            return None
        return self.resample

    def resolve_strategy(self) -> str:
        """Return the canonical strategy to run.

        `strategy` takes precedence; the legacy `min_variance` flag is honored as a
        fallback so old callers/UIs keep working.
        """
        if self.strategy == "llm_bl":
            return "llm_bl"
        if self.strategy == "kelly":
            return "kelly"
        if self.strategy == "min_variance":
            return "min_variance"
        if self.min_variance:
            return "min_variance"
        return "sharpe"

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build Config from a flat dict (e.g. FastAPI form data)."""
        shares = data.get("shares", [])
        if isinstance(shares, str):
            shares = [s.strip() for s in shares.split(",") if s.strip()]

        w_limits_raw = data.get("w_limits", "")
        if isinstance(w_limits_raw, str) and w_limits_raw.strip():
            parts = [float(x.strip()) for x in w_limits_raw.split(",") if x.strip()]
            w_limits = tuple(parts) if len(parts) == 2 else (0.02, 0.12)
        else:
            w_limits = (0.02, 0.12)

        w_limits_per_ticker: dict = {}
        raw_per = data.get("w_limits_per_ticker", "")
        if isinstance(raw_per, str) and raw_per.strip():
            for part in raw_per.split(";"):
                part = part.strip()
                if not part or ":" not in part:
                    continue
                ticker, bounds = part.split(":", 1)
                ticker = ticker.strip().upper()
                if not ticker:
                    continue
                vals = bounds.split(",")
                lo = _parse_limit_side(vals[0])
                hi = _parse_limit_side(vals[1]) if len(vals) > 1 else None
                w_limits_per_ticker[ticker] = (lo, hi)

        min_variance = data.get("min_variance", False) in (True, "true", "on", 1)

        strategy = str(data.get("strategy", "sharpe")).strip().lower()
        if strategy not in ("sharpe", "min_variance", "kelly", "llm_bl"):
            strategy = "sharpe"
        if min_variance and strategy == "sharpe":
            strategy = "min_variance"

        kelly_raw = data.get("kelly_fraction", "")
        try:
            kelly_fraction = float(kelly_raw) if kelly_raw not in (None, "") else 0.5
        except (TypeError, ValueError):
            kelly_fraction = 0.5
        kelly_fraction = min(max(kelly_fraction, 0.01), 1.0)

        llm_samples = _parse_int(data.get("llm_samples", ""), 5, 1, 50)
        llm_tau = _parse_float(data.get("llm_tau", ""), 1.0 / 26.0, 1e-6, 1.0)

        blm_prior = str(data.get("blm_prior", "equal")).strip().lower()
        if blm_prior not in ("equal", "inverse_vol"):
            blm_prior = "equal"

        blm_horizon_days = _parse_int(data.get("blm_horizon_days", ""), 14, 1, 252)
        blm_history_days = _parse_int(data.get("blm_history_days", ""), 56, 5, 504)

        llm_view_engine = str(data.get("llm_view_engine", "agent")).strip().lower()
        if llm_view_engine not in ("agent", "direct"):
            llm_view_engine = "direct"

        llm_reasoning_effort = str(data.get("llm_reasoning_effort", "medium")).strip().lower()
        if llm_reasoning_effort not in ("none", "low", "medium", "high", "xhigh"):
            llm_reasoning_effort = "medium"

        llm_agent_websearch = data.get("llm_agent_websearch", True) in (True, "true", "on", "1", 1)
        llm_agent_max_tokens = _parse_int(data.get("llm_agent_max_tokens", ""), 32768, 512, 32768)
        llm_agent_recursion_limit = _parse_int(data.get("llm_agent_recursion_limit", ""), 40, 5, 200)

        return cls(
            resample=data.get("resample", "none"),
            days=int(data.get("days", 720)),
            shares=shares,
            w_limits=w_limits,
            min_variance=min_variance,
            strategy=strategy,
            kelly_fraction=kelly_fraction,
            llm_samples=llm_samples,
            llm_tau=llm_tau,
            blm_prior=blm_prior,
            blm_horizon_days=blm_horizon_days,
            blm_history_days=blm_history_days,
            llm_view_engine=llm_view_engine,
            llm_reasoning_effort=llm_reasoning_effort,
            llm_agent_websearch=llm_agent_websearch,
            llm_agent_max_tokens=llm_agent_max_tokens,
            llm_agent_recursion_limit=llm_agent_recursion_limit,
            w_limits_per_ticker=w_limits_per_ticker,
            monto_usd=float(data.get("monto_usd", 10000)),
            monthly_delta=float(data.get("monthly_delta", 300)),
            sim_days=int(data.get("sim_days", 252)),
            risk_free_annual_perc=float(data.get("risk_free_annual_perc", 5)),
        )
