"""Shared fixtures for the test suite."""

import numpy as np
import pytest


@pytest.fixture
def mean_ret():
    return np.array([0.001, 0.002, 0.0015])


@pytest.fixture
def cov_ret():
    return np.array([
        [0.0004, 0.0001, 0.00005],
        [0.0001, 0.0009, 0.0002],
        [0.00005, 0.0002, 0.0006],
    ])


@pytest.fixture
def two_mean():
    return np.array([0.001, 0.002])


@pytest.fixture
def two_cov():
    return np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0009],
    ])
