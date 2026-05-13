"""Portfolio optimization configuration."""

from dataclasses import dataclass, field
from typing import List


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
    ])
    w_limits: tuple = (0.02, 0.12)  # (min_weight, max_weight)

    # ── Optimization ─────────────────────────────────────────────────
    min_variance: bool = False  # True = Min-Variance, False = Max Sharpe

    # ── Investment ────────────────────────────────────────────────────
    monto_usd: float = 10000.0
    monthly_delta: float = 300.0

    # ── Simulation ────────────────────────────────────────────────────
    sim_days: int = 252  # trading days for simulation (~1 year)
    risk_free_annual_perc: float = 5.0

    # ── Computed ──────────────────────────────────────────────────────
    @property
    def risk_free(self) -> float:
        """Daily risk-free rate derived from annual percentage."""
        return (1 + self.risk_free_annual_perc / 100) ** (1 / 365) - 1

    def get_resample(self) -> str | None:
        """Return resample string; 'none' → None."""
        if self.resample == "none":
            return None
        return self.resample

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

        return cls(
            resample=data.get("resample", "none"),
            days=int(data.get("days", 720)),
            shares=shares,
            w_limits=w_limits,
            min_variance=data.get("min_variance", False) in (True, "true", "on", 1),
            monto_usd=float(data.get("monto_usd", 10000)),
            monthly_delta=float(data.get("monthly_delta", 300)),
            sim_days=int(data.get("sim_days", 252)),
            risk_free_annual_perc=float(data.get("risk_free_annual_perc", 5)),
        )
