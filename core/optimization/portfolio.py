"""Portfolio math: return and risk metrics from weights."""

import numpy as np


def port_mean(W: np.ndarray, R: np.ndarray) -> float:
    """Calculate portfolio mean return."""
    return float(np.sum(R * W))


def port_var(W: np.ndarray, C: np.ndarray) -> float:
    """Calculate portfolio variance of returns."""
    return float(np.dot(np.dot(W, C), W))


def port_mean_var(W: np.ndarray, R: np.ndarray, C: np.ndarray) -> tuple:
    """Calculate portfolio mean return and variance."""
    return port_mean(W, R), port_var(W, C)
