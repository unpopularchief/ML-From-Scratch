"""Input validation shared by every estimator.

These are plumbing, not algorithms: they turn whatever the caller passes in
into the single array shape/dtype the rest of the codebase can rely on
(float64, C-contiguous, no NaN/inf), and raise early with a clear message
instead of letting a bad shape surface as a confusing error three functions
later.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.exceptions import NotFittedError
from scratchgrad.typing import FeatureMatrix, TargetVector


def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Turn a seed into a ``numpy.random.Generator``.

    The project never touches NumPy's global RNG (``np.random.seed`` /
    bare ``np.random.rand``) — every estimator takes an explicit
    ``random_state`` and threads it through as a ``Generator`` instead, so
    two runs with the same seed are reproducible regardless of what else
    ran before them.

    Parameters
    ----------
    seed : int, numpy.random.Generator, or None
        If ``None``, a fresh, unseeded ``Generator`` is returned (calls
        will differ on every run). If an ``int``, it seeds a new
        ``Generator``. If already a ``Generator``, it is returned as-is
        so callers can share one generator across several estimators.

    Returns
    -------
    numpy.random.Generator

    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, int):
        return np.random.default_rng(seed)
    raise TypeError(
        f"random_state must be None, an int, or a numpy.random.Generator, "
        f"got {type(seed).__name__!r}."
    )


def check_X_y(X: object, y: object) -> tuple[FeatureMatrix, TargetVector]:
    """Validate and coerce a design matrix and target vector.

    Enforces the project-wide contract used everywhere else in the
    codebase: ``X`` is a 2D float64 array of shape ``(n_samples,
    n_features)``, ``y`` is a 1D float64 array of shape ``(n_samples,)``,
    both finite, with matching ``n_samples``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like of shape (n_samples,)
        Target values (labels or regression targets).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features), dtype float64
    y : ndarray of shape (n_samples,), dtype float64

    Raises
    ------
    ValueError
        If ``X`` is not 2D, ``y`` is not 1D, their sample counts disagree,
        either contains NaN/inf, or ``n_samples == 0``.

    """
    X_arr = check_array(X)
    y_arr = np.asarray(y, dtype=np.float64)

    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains NaN or infinite values.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y have mismatched n_samples: {X_arr.shape[0]} != {y_arr.shape[0]}."
        )
    return X_arr, y_arr


def check_array(X: object) -> FeatureMatrix:
    """Validate and coerce a single 2D array (no target vector).

    Used for ``predict``/``transform`` inputs, where there is no ``y`` to
    validate alongside ``X``. See :func:`check_X_y` for the full contract.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to validate and coerce.

    Returns
    -------
    ndarray of shape (n_samples, n_features), dtype float64

    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X_arr.shape}.")
    if X_arr.shape[0] == 0:
        raise ValueError("X has 0 samples.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains NaN or infinite values.")
    return X_arr


def check_sample_weight(sample_weight: object, n_samples: int) -> np.ndarray:
    """Validate per-sample weights, or synthesise uniform ones.

    Several estimators (``DecisionTreeClassifier``, and the boosting
    ensembles that reweight it) accept a ``sample_weight`` vector that
    scales each row's contribution to the fit. This turns whatever the
    caller passes into a length-``n_samples`` ``float64`` array, or returns
    all-ones when it is ``None``.

    Parameters
    ----------
    sample_weight : array-like of shape (n_samples,) or None
        Non-negative weights, not all zero. ``None`` yields ``np.ones``.
    n_samples : int
        The number of rows the weights must align with.

    Returns
    -------
    ndarray of shape (n_samples,), dtype float64

    Raises
    ------
    ValueError
        If ``sample_weight`` is not 1D of length ``n_samples``, contains
        NaN/inf or a negative entry, or sums to zero.

    """
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (n_samples,):
        raise ValueError(
            f"sample_weight must have shape ({n_samples},), got {w.shape}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight contains NaN or infinite values.")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be non-negative.")
    if w.sum() == 0.0:
        raise ValueError("sample_weight sums to zero.")
    return w


def check_is_fitted(estimator: object, attributes: str | list[str]) -> None:
    """Raise ``NotFittedError`` unless the given learned attribute(s) exist.

    Parameters
    ----------
    estimator : object
        The estimator instance to check.
    attributes : str or list of str
        Name(s) of trailing-underscore attribute(s) that ``fit`` sets
        (e.g. ``"coef_"``). Checked with ``hasattr``.

    Raises
    ------
    NotFittedError
        If any of ``attributes`` is missing from ``estimator``.

    """
    names = [attributes] if isinstance(attributes, str) else attributes
    if not all(hasattr(estimator, name) for name in names):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            f"Call 'fit' before using this estimator."
        )
