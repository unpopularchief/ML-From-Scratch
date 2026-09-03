"""Shared pytest fixtures and tolerance constants.

Every test that needs randomness uses the ``rng`` fixture below rather
than calling ``np.random.default_rng`` directly, so a single seed change
here reproduces (or changes) every test's data at once.
"""

from __future__ import annotations

import numpy as np
import pytest

# Fixed seed: deterministic across runs and machines. Not meant to be
# changed casually — a flaky test that only fails for some seeds is a
# real bug (e.g. an edge case in the synthetic data), not a reason to
# reroll until it passes.
SEED = 0

# Absolute/relative tolerance for analytic correctness checks (tier 1,
# plan.md section 3), e.g. comparing to a closed-form solution.
ATOL = 1e-8
RTOL = 1e-6


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded ``numpy.random.Generator`` for generating test data."""
    return np.random.default_rng(SEED)
