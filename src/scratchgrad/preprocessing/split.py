"""Train/test splitting."""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FeatureMatrix, TargetVector
from scratchgrad.utils.validation import check_random_state


def train_test_split(
    X: FeatureMatrix,
    y: TargetVector,
    test_size: float = 0.25,
    random_state: int | np.random.Generator | None = None,
) -> tuple[FeatureMatrix, FeatureMatrix, TargetVector, TargetVector]:
    """Randomly split ``X``/``y`` into training and test sets.

    Shuffles sample indices, then takes the first ``round(n_samples *
    test_size)`` of the shuffled order as the test set and the rest as
    train — so both splits are random subsets of the original rows, not
    biased toward either end of the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix to split.
    y : ndarray of shape (n_samples,)
        Targets to split, paired row-for-row with ``X``.
    test_size : float, default=0.25
        Fraction of samples to allocate to the test set, in ``(0, 1)``.
    random_state : int, numpy.random.Generator, or None
        Seed controlling the shuffle. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        ``X_train``/``y_train`` have ``n_samples - n_test`` rows;
        ``X_test``/``y_test`` have ``n_test`` rows, where
        ``n_test = round(n_samples * test_size)``.

    Raises
    ------
    ValueError
        If ``test_size`` is not in ``(0, 1)``, or if the split would leave
        the train or test set with 0 samples.

    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    n_samples = X.shape[0]
    n_test = round(n_samples * test_size)
    if n_test == 0 or n_test == n_samples:
        raise ValueError(
            f"test_size={test_size} on {n_samples} samples leaves an empty "
            f"train or test set."
        )

    rng = check_random_state(random_state)
    shuffled_indices = rng.permutation(n_samples)
    test_indices, train_indices = shuffled_indices[:n_test], shuffled_indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
