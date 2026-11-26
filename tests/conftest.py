"""Shared test configuration and deterministic random seeds."""

from collections.abc import Iterator

import numpy as np
import pytest
import torch as T


@pytest.fixture(autouse=True)
def deterministic_rngs() -> Iterator[None]:
    """Reset NumPy and PyTorch RNGs before each test."""
    T.manual_seed(0)
    np.random.seed(0)
    yield
