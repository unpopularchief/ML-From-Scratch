"""Validation and numerical helpers shared across the codebase."""

from scratchgrad.utils.math import logsumexp, sigmoid, softmax
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_sample_weight,
    check_X_y,
)

__all__ = [
    "check_X_y",
    "check_array",
    "check_is_fitted",
    "check_random_state",
    "check_sample_weight",
    "logsumexp",
    "sigmoid",
    "softmax",
]
