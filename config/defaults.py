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
    strategy: str = "sharpe"  # "sharpe" | "min_variance" | "kelly"
    kelly_fraction: float = 0.5  # Kelly aggressiveness λ: 1.0 = full Kelly, 0.5 = half-Kelly (recommended)

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
        if strategy not in ("sharpe", "min_variance", "kelly"):
            strategy = "sharpe"
        if min_variance and strategy == "sharpe":
            strategy = "min_variance"

        kelly_raw = data.get("kelly_fraction", "")
        try:
            kelly_fraction = float(kelly_raw) if kelly_raw not in (None, "") else 0.5
        except (TypeError, ValueError):
            kelly_fraction = 0.5
        kelly_fraction = min(max(kelly_fraction, 0.01), 1.0)

        return cls(
            resample=data.get("resample", "none"),
            days=int(data.get("days", 720)),
            shares=shares,
            w_limits=w_limits,
            min_variance=min_variance,
            strategy=strategy,
            kelly_fraction=kelly_fraction,
            w_limits_per_ticker=w_limits_per_ticker,
            monto_usd=float(data.get("monto_usd", 10000)),
            monthly_delta=float(data.get("monthly_delta", 300)),
            sim_days=int(data.get("sim_days", 252)),
            risk_free_annual_perc=float(data.get("risk_free_annual_perc", 5)),
        )
