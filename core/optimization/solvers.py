"""Portfolio optimization: Mean-Variance (Sharpe), Minimal Variance and Kelly Growth solvers."""

import numpy as np
import scipy.optimize
from cvxopt.solvers import qp, options as cvxopt_options
from cvxopt import matrix

from core.optimization.portfolio import port_mean_var


def _asset_bounds(w_limits, n: int) -> list:
    """Normalize bounds: per-asset pairs or one pair broadcast to all assets."""
    if isinstance(w_limits, list) and len(w_limits) == n and all(
        isinstance(bound, (list, tuple, np.ndarray)) and len(bound) == 2
        for bound in w_limits
    ):
        return [tuple(bound) for bound in w_limits]
    if not isinstance(w_limits, (list, tuple, np.ndarray)) or len(w_limits) != 2:
        raise ValueError("weight limits must be a (min, max) pair or one pair per asset")
    return [tuple(w_limits)] * n


def _validate_bounds(bounds: list) -> None:
    if not bounds:
        raise ValueError("at least one asset is required")
    for i, bound in enumerate(bounds):
        if len(bound) != 2:
            raise ValueError(f"weight bounds for asset {i} must contain min and max")
        lo, hi = (float(bound[0]), float(bound[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi > 1 or lo > hi:
            raise ValueError(
                f"invalid weight bounds for asset {i}: expected 0 <= min <= max <= 1"
            )


def _assert_bounds_feasible(bounds: list) -> None:
    """Weights must sum to 1: the bounds must leave a non-empty feasible set."""
    _validate_bounds(bounds)
    lo = sum(float(b[0]) for b in bounds)
    hi = sum(float(b[1]) for b in bounds)
    if hi < 1.0 - 1e-9:
        raise ValueError(
            f"weight bounds infeasible: with {len(bounds)} assets the max weights sum to "
            f"{hi:.2f} < 1.0 — raise the max weight or add more tickers"
        )
    if lo > 1.0 + 1e-9:
        raise ValueError(
            f"weight bounds infeasible: the min weights sum to {lo:.2f} > 1.0 — lower the min weight"
        )


def _validate_covariance(cov_returns: np.ndarray, n: int) -> np.ndarray:
    cov = np.asarray(cov_returns, dtype=float)
    if cov.shape != (n, n):
        raise ValueError(f"covariance matrix must have shape ({n}, {n})")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance matrix must contain only finite values")
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues[0] < -1e-8:
        raise ValueError("covariance matrix must be positive semidefinite")
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T


def _feasible_start(bounds: list) -> np.ndarray:
    lower = np.array([float(b[0]) for b in bounds])
    upper = np.array([float(b[1]) for b in bounds])
    weights = lower.copy()
    remaining = 1.0 - float(weights.sum())
    for i in range(len(weights)):
        amount = min(remaining, upper[i] - weights[i])
        weights[i] += amount
        remaining -= amount
    if remaining > 1e-9:
        raise ValueError("weight bounds do not leave a feasible portfolio")
    return weights


def _validate_solution(weights: np.ndarray, bounds: list) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)) or abs(float(weights.sum()) - 1.0) > 1e-6:
        raise RuntimeError("optimizer returned invalid portfolio weights")
    for weight, (lo, hi) in zip(weights, bounds):
        if weight < float(lo) - 1e-6 or weight > float(hi) + 1e-6:
            raise RuntimeError("optimizer returned weights outside the requested bounds")
    return weights


def solve_mean_variance(
    mean_returns: np.ndarray,
    cov_returns: np.ndarray,
    rf: float,
    w_limits: tuple | list,
) -> np.ndarray:
    """Mean-Variance optimization: maximize Sharpe ratio.

    Args:
        mean_returns: Array of asset mean returns.
        cov_returns:  Covariance matrix of returns.
        rf:           Daily risk-free rate.
        w_limits:     Single (min_weight, max_weight) tuple, or a list of
                      per-asset (min_weight, max_weight) tuples.

    Returns:
        weights: Optimized portfolio weights.
    """
    mean_returns = np.asarray(mean_returns, dtype=float).reshape(-1)
    n = len(mean_returns)
    if n == 0 or not np.all(np.isfinite(mean_returns)):
        raise ValueError("mean returns must be a non-empty finite vector")
    cov_returns = _validate_covariance(cov_returns, n)
    bounds = _asset_bounds(w_limits, n)
    _assert_bounds_feasible(bounds)

    def fitness(W: np.ndarray, R: np.ndarray, C: np.ndarray, rf_val: float) -> float:
        mean, var = port_mean_var(W, R, C)
        if not np.isfinite(mean) or not np.isfinite(var) or var < 0:
            return 1e12
        if var <= 1e-15:
            return -1e6 if mean > rf_val else 1e6
        return -(mean - rf_val) / np.sqrt(var)

    W0 = _feasible_start(bounds)
    constraints = {"type": "eq", "fun": lambda W: np.sum(W) - 1.0}

    result = scipy.optimize.minimize(
        fitness, W0, (mean_returns, cov_returns, rf),
        method="SLSQP", constraints=constraints, bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return _validate_solution(result.x, bounds)


def solve_min_variance(
    cov_returns: np.ndarray,
    w_limits: tuple | list,
    n_assets: int,
) -> np.ndarray:
    """Minimal Variance optimization via quadratic programming.

    Args:
        cov_returns: Covariance matrix of returns.
        w_limits:    Single (min_weight, max_weight) tuple, or a list of
                     per-asset (min_weight, max_weight) tuples.
        n_assets:    Number of assets.

    Returns:
        weights: Optimized portfolio weights.
    """
    cvxopt_options["show_progress"] = False

    if n_assets <= 0:
        raise ValueError("at least one asset is required")
    bounds = _asset_bounds(w_limits, n_assets)
    _assert_bounds_feasible(bounds)
    cov_returns = _validate_covariance(cov_returns, n_assets)
    low_up_bound = [-b[0] for b in bounds] + [b[1] for b in bounds]

    P = matrix(cov_returns)
    q = matrix(0.0, (n_assets, 1))
    G = matrix(np.append(
        np.diag([-1.0] * n_assets),
        np.diag([1.0] * n_assets), 0,
    ))
    h = matrix(np.array([[float(v)] for v in low_up_bound]))
    A = matrix(1.0, (1, n_assets))
    b = matrix(1.0)

    try:
        sol = qp(P, q, G, h, A, b)
    except Exception as exc:
        raise RuntimeError(f"minimal variance optimization failed: {exc}") from exc
    if sol.get("status") != "optimal":
        raise RuntimeError(f"minimal variance optimization failed: {sol.get('status')}")
    return _validate_solution(np.array(sol["x"]).flatten(), bounds)


def solve_kelly_growth(
    mean_returns: np.ndarray,
    cov_returns: np.ndarray,
    kelly_fraction: float,
    w_limits: tuple | list,
) -> np.ndarray:
    """Kelly / growth-optimal optimization: maximize the long-term geometric growth rate.

    Maximizes `w'μ - (1/(2λ)) w'Σw` — the mean-variance objective with the risk penalty
    fixed by the Kelly fraction λ. λ=1 is full Kelly (aggressive); λ<1 (e.g. 0.5 half-Kelly)
    is fractional Kelly, which tames drawdowns and estimation-error sensitivity while
    retaining most of the growth.

    Args:
        mean_returns:   Array of asset mean returns.
        cov_returns:    Covariance matrix of returns.
        kelly_fraction: Kelly fraction λ in (0, 1]; 1.0 = full Kelly.
        w_limits:       Single (min_weight, max_weight) tuple, or a list of
                        per-asset (min_weight, max_weight) tuples.

    Returns:
        weights: Optimized portfolio weights.
    """
    cvxopt_options["show_progress"] = False

    mean_returns = np.asarray(mean_returns, dtype=float).reshape(-1)
    n = len(mean_returns)
    if n == 0 or not np.all(np.isfinite(mean_returns)):
        raise ValueError("mean returns must be a non-empty finite vector")
    lam = min(max(kelly_fraction, 0.01), 1.0)

    bounds = _asset_bounds(w_limits, n)
    _assert_bounds_feasible(bounds)
    cov_returns = _validate_covariance(cov_returns, n)
    low_up_bound = [-b[0] for b in bounds] + [b[1] for b in bounds]

    P = matrix((1.0 / lam) * cov_returns)
    q = matrix(-mean_returns.reshape(n, 1))
    G = matrix(np.append(
        np.diag([-1.0] * n),
        np.diag([1.0] * n), 0,
    ))
    h = matrix(np.array([[float(v)] for v in low_up_bound]))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = qp(P, q, G, h, A, b)
    if sol["status"] != "optimal":
        raise RuntimeError(f"Kelly growth optimization failed: {sol['status']}")
    return _validate_solution(np.array(sol["x"]).flatten(), bounds)


def optimize(
    mean_returns: np.ndarray,
    cov_returns: np.ndarray,
    rf: float,
    w_limits: tuple | list,
    min_variance: bool = False,
    strategy: str = "sharpe",
    kelly_fraction: float = 0.5,
) -> tuple:
    """Run the selected optimization strategy.

    Args:
        mean_returns: Array of asset mean returns.
        cov_returns:  Covariance matrix of returns.
        rf:           Daily risk-free rate.
        w_limits:     Single (min_weight, max_weight) tuple, or a list of
                      per-asset (min_weight, max_weight) tuples.
        min_variance: Legacy flag (backward compat); `strategy` takes precedence.
        strategy:     "sharpe" | "min_variance" | "kelly".
        kelly_fraction: Kelly fraction λ used when strategy is "kelly".

    Returns:
        (weights, mean, std, strategy_name)
    """
    mean_returns = np.asarray(mean_returns, dtype=float).reshape(-1)
    n = len(mean_returns)
    if n == 0 or not np.all(np.isfinite(mean_returns)):
        raise ValueError("mean returns must be a non-empty finite vector")
    cov_returns = _validate_covariance(cov_returns, n)
    bounds = _asset_bounds(w_limits, n)
    _assert_bounds_feasible(bounds)

    if strategy == "kelly":
        weights = solve_kelly_growth(mean_returns, cov_returns, kelly_fraction, bounds)
        strategy_name = f"Growth Optimization (Kelly {round(kelly_fraction * 100)}%)"
    elif strategy == "min_variance" or min_variance:
        weights = solve_min_variance(cov_returns, bounds, n)
        strategy_name = "Minimal Variance Optimization"
    else:
        weights = solve_mean_variance(mean_returns, cov_returns, rf, bounds)
        strategy_name = "Mean-Variance Optimization (historical)"

    mean, var = port_mean_var(weights, mean_returns, cov_returns)
    std = np.sqrt(var)
    return weights, mean, std, strategy_name
