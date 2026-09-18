"""LLM-Black-Litterman: LLM return views blended with an equilibrium prior.

Implements the LLM -> Black-Litterman framework (Lee et al., arXiv:2504.14345)
with production-oriented improvements over the reference implementation:

- JSON-first response parsing with a plain-number fallback
- Regularized confidence: omega_ii = max(view_var, floor) with
  floor = OMEGA_FLOOR_FRACTION * tau * Sigma_ii, so an overconfident LLM
  (near-zero sample variance) can neither dominate the prior nor make the
  confidence matrix numerically singular
- Partial views: assets without valid samples are excluded from the picking
  matrix instead of failing the run
- Graceful degradation: when the LLM is unreachable or every view is
  unparseable, the strategy falls back to the pure equilibrium prior
- No market-cap requirement: the equilibrium prior uses equal or
  inverse-volatility weights
- Daily view caching: same-day runs with identical inputs reuse the cached
  LLM views (reproducible, zero extra API cost)
- Thinking-model handling: vLLM-served reasoning models (e.g. Qwen3) would
  otherwise burn the whole token budget "thinking" and return empty content.
  View calls therefore send chat_template_kwargs={"enable_thinking": false}
  (override with LLM_ENABLE_THINKING=1) and retry without it once when the
  endpoint rejects the parameter
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.defaults import Config
from config.llm import LLMSettings
from core.data import cache as price_cache
from core.data.loader import ticker_for
from models.llm_client import chat_completion

OMEGA_FLOOR_FRACTION = 0.25
VIEW_TEMPERATURE = 0.8
MIN_HISTORY_OBS = 10
MAX_PLAUSIBLE_DAILY_PCT = 50.0
VIEW_MAX_TOKENS_FAST = 512
VIEW_MAX_TOKENS_THINKING = 2048
_EXTRABODY_SUPPORT: dict = {"supports": None}

SYSTEM_PROMPT = (
    "You are a professional quant equity analyst. You receive an asset's recent "
    "daily simple returns in percent per day (e.g. 0.23 means +0.23% that day). "
    "Forecast the AVERAGE daily return for the next {horizon} trading days, in the "
    "same percent-per-day units, based on the provided history plus general market "
    "knowledge. Answer with ONLY one JSON object: "
    '{{"predicted_avg_daily_return_pct": <number>}}'
)


def build_equilibrium_prior(
    cov_returns: np.ndarray,
    tau: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reverse-optimized equilibrium expected returns: pi = (tau * Sigma) @ w."""
    cov = np.asarray(cov_returns, dtype=float)
    if weights is None:
        weights = np.ones(len(cov))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return tau * (cov @ weights)


def inverse_vol_weights(cov_returns: np.ndarray) -> np.ndarray:
    """Risk-budgeting weights: w_i proportional to 1/vol_i, normalized to sum 1."""
    vols = np.sqrt(np.diag(np.asarray(cov_returns, dtype=float)))
    inv = 1.0 / vols
    return inv / inv.sum()


def _solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix) @ vector
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ vector


def black_litterman_posterior(
    prior_mean: np.ndarray,
    cov_returns: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Posterior expected returns.

    mu_BL = [A^-1 + P'Omega^-1P]^-1 [A^-1 pi + P'Omega^-1q]
    with A = tau * Sigma.

    Args:
        prior_mean: Equilibrium expected returns (pi).
        cov_returns: Covariance matrix (same frequency units as the views).
        P:           (r, n) picking matrix; each row selects one absolute view.
        q:           (r,) view expected returns.
        omega:       (r, r) view confidence matrix (diagonal in this module).
        tau:         Prior covariance scaling.
    """
    prior = np.asarray(prior_mean, dtype=float)
    cov = np.asarray(cov_returns, dtype=float)
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    omega = np.asarray(omega, dtype=float)

    prior_inv = np.linalg.pinv(tau * cov + 1e-10 * np.eye(len(cov)))
    omega_inv = np.linalg.pinv(omega + 1e-10 * np.eye(len(omega)))
    A = prior_inv + P.T @ omega_inv @ P
    b = prior_inv @ prior + P.T @ omega_inv @ q
    return _solve(A, b)


def _parse_view_response(text: str) -> float | None:
    """Extract a predicted average daily return (percent per day) from an LLM response."""
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            for key in ("predicted_avg_daily_return_pct", "avg_daily_return_pct", "value"):
                if key in obj:
                    return float(obj[key])
        except (ValueError, TypeError):
            pass
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None


def _build_prompt(name: str, cfg: Config, history_pct: np.ndarray) -> list[dict]:
    symbol = ticker_for(name)
    if symbol:
        ident = f"ETF {name} (display ticker: {symbol})"
    else:
        ident = f"Stock {name}"
    hist = ", ".join(f"{v:.2f}" for v in history_pct)
    user = (
        f"Asset: {ident}\n"
        f"Past {len(history_pct)} daily returns, percent per day, oldest first: [{hist}]\n"
        f"Forecast the average daily return (percent per day) for the next "
        f"{cfg.blm_horizon_days} trading days."
    )
    system = SYSTEM_PROMPT.format(horizon=cfg.blm_horizon_days)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _views_cache_path(names: list[str], model: str, cfg: Config) -> str:
    key_src = "|".join([
        model,
        str(cfg.llm_samples),
        str(cfg.llm_tau),
        str(cfg.blm_prior),
        str(cfg.blm_horizon_days),
        str(cfg.blm_history_days),
        ",".join(names),
    ])
    digest = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{price_cache.CACHE_DIR}/llm_views_{digest}_{today}.json"


def _read_cached_views(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        views = data.get("views")
        return views if isinstance(views, dict) else None
    except (OSError, ValueError):
        return None


def _write_cached_views(path: str, model: str, views: dict) -> None:
    os.makedirs(price_cache.CACHE_DIR, exist_ok=True)
    payload = {
        "model": model,
        "views": {
            name: {
                "q": v["q"],
                "omega": v["omega"],
                "samples": v["samples"],
                "n_ok": v["n_ok"],
            }
            for name, v in views.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def _view_extra_body() -> dict | None:
    """Disable reasoning-model thinking for fast, direct view answers.

    Returns None when the user forces thinking on (LLM_ENABLE_THINKING=1) or when
    a previous probe found the endpoint does not accept chat_template_kwargs.
    """
    if os.environ.get("LLM_ENABLE_THINKING", "").strip().lower() in ("1", "true", "yes"):
        return None
    if _EXTRABODY_SUPPORT["supports"] is False:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _call_for_sample(prompt: list[dict], settings: LLMSettings) -> str | None:
    """One LLM sample with a one-shot fallback if the endpoint rejects extra body fields."""
    extra = _view_extra_body()
    if extra is not None:
        try:
            return chat_completion(
                prompt, settings=settings,
                temperature=VIEW_TEMPERATURE, max_tokens=VIEW_MAX_TOKENS_FAST, extra_body=extra,
            )
        except RuntimeError:
            _EXTRABODY_SUPPORT["supports"] = False
    return chat_completion(
        prompt, settings=settings,
        temperature=VIEW_TEMPERATURE, max_tokens=VIEW_MAX_TOKENS_THINKING,
    )


def generate_views(
    returns: pd.DataFrame,
    names: list[str],
    cov_returns: np.ndarray,
    cfg: Config,
    settings: LLMSettings | None = None,
    force_refresh: bool = False,
) -> tuple[dict, bool]:
    """Generate per-asset LLM views with confidence from K samples.

    Args:
        returns: Daily simple-returns DataFrame (index = dates, one column per asset).
        names:   Asset names (columns of `returns`).
        cov_returns: Daily covariance matrix (for the confidence floor).
        cfg:     Pipeline config (samples, tau, horizon, history window).
        settings: LLM endpoint settings; defaults to LLMSettings.from_env().
        force_refresh: Ignore today's cache and re-query the LLM.

    Returns:
        (views, from_cache) — views maps name -> {q, omega, samples, n_ok},
        with q/omega in decimal daily-return units.
    """
    if settings is None:
        settings = LLMSettings.from_env()

    path = _views_cache_path(names, settings.model, cfg)
    cached = None if force_refresh else _read_cached_views(path)
    if cached is not None:
        return cached, True

    cov_diag = np.diag(np.asarray(cov_returns, dtype=float))
    omega_floor = OMEGA_FLOOR_FRACTION * cfg.llm_tau * cov_diag
    returns_asc = returns.sort_index()

    views: dict = {}
    for j, name in enumerate(names):
        series = returns_asc[name].dropna()
        if len(series) < MIN_HISTORY_OBS:
            continue
        history_pct = series.tail(cfg.blm_history_days).to_numpy(dtype=float) * 100.0
        prompt = _build_prompt(name, cfg, history_pct)

        samples: list[float] = []
        for _ in range(cfg.llm_samples):
            try:
                raw = _call_for_sample(prompt, settings)
            except RuntimeError:
                break
            value = _parse_view_response(raw)
            if value is None or not np.isfinite(value) or abs(value) > MAX_PLAUSIBLE_DAILY_PCT:
                continue
            samples.append(value / 100.0)
        if not samples:
            continue

        q = float(np.mean(samples))
        var = float(np.var(samples, ddof=1)) if len(samples) > 1 else float(omega_floor[j])
        omega = max(var, float(omega_floor[j]))
        views[name] = {
            "q": q,
            "omega": omega,
            "samples": [float(s) for s in samples],
            "n_ok": len(samples),
        }

    _write_cached_views(path, settings.model, views)
    return views, False


def run_blm(
    returns: pd.DataFrame,
    names: list[str],
    cov_returns: np.ndarray,
    cfg: Config,
    settings: LLMSettings | None = None,
) -> dict:
    """Run the full LLM-Black-Litterman flow on daily-frequency data.

    Returns:
        {posterior_mean, prior_mean, views, view_names, from_cache, degraded, model}
    """
    if settings is None:
        settings = LLMSettings.from_env()

    cov = np.asarray(cov_returns, dtype=float)
    weights = inverse_vol_weights(cov) if cfg.blm_prior == "inverse_vol" else None
    prior = build_equilibrium_prior(cov, cfg.llm_tau, weights)

    views: dict = {}
    from_cache = False
    try:
        views, from_cache = generate_views(returns, names, cov, cfg, settings=settings)
    except Exception:
        views, from_cache = {}, False

    view_names = [n for n in names if n in views]
    if view_names:
        idx = [names.index(n) for n in view_names]
        P = np.zeros((len(view_names), len(names)))
        for row, col in enumerate(idx):
            P[row, col] = 1.0
        q = np.array([views[n]["q"] for n in view_names])
        omega = np.diag([views[n]["omega"] for n in view_names])
        posterior = black_litterman_posterior(prior, cov, P, q, omega, cfg.llm_tau)
        degraded = False
    else:
        posterior = prior
        degraded = True

    return {
        "posterior_mean": posterior,
        "prior_mean": prior,
        "views": views,
        "view_names": view_names,
        "from_cache": from_cache,
        "degraded": degraded,
        "model": settings.model,
    }
