"""Portfolio optimization: Mean-Variance (Sharpe) and Minimal Variance solvers."""

import numpy as np
import scipy.optimize
from cvxopt.solvers import qp, options as cvxopt_options
from cvxopt import matrix


def port_mean(W: np.ndarray, R: np.ndarray) -> float:
    """Calculate portfolio mean return."""
    return float(np.sum(R * W))


def port_var(W: np.ndarray, C: np.ndarray) -> float:
    """Calculate portfolio variance of returns."""
    return float(np.dot(np.dot(W, C), W))


def port_mean_var(W: np.ndarray, R: np.ndarray, C: np.ndarray) -> tuple:
    """Calculate portfolio mean return and variance."""
    return port_mean(W, R), port_var(W, C)


def solve_mean_variance(
    mean_returns: np.ndarray,
    cov_returns: np.ndarray,
    rf: float,
    w_limits: tuple,
) -> np.ndarray:
    """Mean-Variance optimization: maximize Sharpe ratio.

    Args:
        mean_returns: Array of asset mean returns.
        cov_returns:  Covariance matrix of returns.
        rf:           Daily risk-free rate.
        w_limits:     Tuple (min_weight, max_weight) per asset.

    Returns:
        weights: Optimized portfolio weights.
    """
    def fitness(W: np.ndarray, R: np.ndarray, C: np.ndarray, rf_val: float) -> float:
        mean, var = port_mean_var(W, R, C)
        sharpe = (mean - rf_val) / np.sqrt(var)
        return 1.0 / sharpe  # minimize inverse Sharpe

    n = len(mean_returns)
    W0 = np.ones(n) / n
    bounds = [w_limits for _ in range(n)]
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
    w_limits: tuple,
    n_assets: int,
) -> np.ndarray:
    """Minimal Variance optimization via quadratic programming.

    Args:
        cov_returns: Covariance matrix of returns.
        w_limits:    Tuple (min_weight, max_weight) per asset.
        n_assets:    Number of assets.

    Returns:
        weights: Optimized portfolio weights.
    """
    cvxopt_options["show_progress"] = False

    low_up_bound = [0.0] * n_assets + [w_limits[1]] * n_assets

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


def optimize(
    mean_returns: np.ndarray,
    cov_returns: np.ndarray,
    rf: float,
    w_limits: tuple,
    min_variance: bool = False,
) -> tuple:
    """Run the selected optimization strategy.

    Args:
        mean_returns: Array of asset mean returns.
        cov_returns:  Covariance matrix of returns.
        rf:           Daily risk-free rate.
        w_limits:     Tuple (min_weight, max_weight) per asset.
        min_variance: If True use Min-Variance, else Mean-Variance (Sharpe).

    Returns:
        (weights, mean, std, strategy_name)
    """
    n = len(mean_returns)

    if min_variance:
        weights = solve_min_variance(cov_returns, w_limits, n)
        strategy = "Minimal Variance Optimization"
    else:
        weights = solve_mean_variance(mean_returns, cov_returns, rf, w_limits)
        strategy = "Mean-Variance Optimization (historical)"

    mean, var = port_mean_var(weights, mean_returns, cov_returns)
    std = np.sqrt(var)
    return weights, mean, std, strategy
