"""Shared type aliases.

The whole project uses a single dtype (float64 — see docs/conventions.md), so
these aliases exist purely for readability at call sites: they say what an
array *means* (a design matrix vs. a label vector) without adding any actual
type-checking machinery.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# A 2D array of shape (n_samples, n_features).
FeatureMatrix = npt.NDArray[np.float64]

# A 1D array of shape (n_samples,) — regression targets or class labels.
TargetVector = npt.NDArray[np.float64]

# A generic float64 array whose shape depends on context (weights, gradients, ...).
FloatArray = npt.NDArray[np.float64]

# An integer array, e.g. class labels or indices.
IntArray = npt.NDArray[np.int64]
