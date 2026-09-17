"""Portfolio optimization: Mean-Variance (Sharpe), Minimal Variance and Kelly Growth solvers."""

import numpy as np
import scipy.optimize
from cvxopt.solvers import qp, options as cvxopt_options
from cvxopt import matrix

from core.optimization.portfolio import port_mean_var


def _asset_bounds(w_limits, n: int) -> list:
    """Normalize bounds: a list of n (min, max) pairs, or a single tuple broadcast to all assets."""
    if isinstance(w_limits, list) and len(w_limits) == n:
        return [tuple(b) for b in w_limits]
    return [tuple(w_limits)] * n


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
    def fitness(W: np.ndarray, R: np.ndarray, C: np.ndarray, rf_val: float) -> float:
        mean, var = port_mean_var(W, R, C)
        sharpe = (mean - rf_val) / np.sqrt(var)
        return 1.0 / sharpe  # minimize inverse Sharpe

    n = len(mean_returns)
    W0 = np.ones(n) / n
    bounds = _asset_bounds(w_limits, n)
    constraints = {"type": "eq", "fun": lambda W: np.sum(W) - 1.0}

    result = scipy.optimize.minimize(
        fitness, W0, (mean_returns, cov_returns, rf),
        method="SLSQP", constraints=constraints, bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


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

    bounds = _asset_bounds(w_limits, n_assets)
    low_up_bound = [-b[0] for b in bounds] + [b[1] for b in bounds]

    P = matrix(np.array(cov_returns, dtype=float))
    q = matrix(0.0, (n_assets, 1))
    G = matrix(np.append(
        np.diag([-1.0] * n_assets),
        np.diag([1.0] * n_assets), 0,
    ))
    h = matrix(np.array([[float(v)] for v in low_up_bound]))
    A = matrix(1.0, (1, n_assets))
    b = matrix(1.0)

    sol = qp(P, q, G, h, A, b)
    return np.array(sol["x"]).flatten()


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

    n = len(mean_returns)
    lam = min(max(kelly_fraction, 0.01), 1.0)

    bounds = _asset_bounds(w_limits, n)
    low_up_bound = [-b[0] for b in bounds] + [b[1] for b in bounds]

    P = matrix((1.0 / lam) * np.array(cov_returns, dtype=float))
    q = matrix(-np.array(mean_returns, dtype=float).reshape(n, 1))
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
    return np.array(sol["x"]).flatten()


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
    n = len(mean_returns)

    if strategy == "kelly":
        weights = solve_kelly_growth(mean_returns, cov_returns, kelly_fraction, w_limits)
        strategy_name = f"Growth Optimization (Kelly {round(kelly_fraction * 100)}%)"
    elif strategy == "min_variance" or min_variance:
        weights = solve_min_variance(cov_returns, w_limits, n)
        strategy_name = "Minimal Variance Optimization"
    else:
        weights = solve_mean_variance(mean_returns, cov_returns, rf, w_limits)
        strategy_name = "Mean-Variance Optimization (historical)"

    mean, var = port_mean_var(weights, mean_returns, cov_returns)
    std = np.sqrt(var)
    return weights, mean, std, strategy_name
