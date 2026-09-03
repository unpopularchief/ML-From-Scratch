"""Feature scaling transformers.

Both follow the ``fit``/``transform`` contract from :mod:`scratchgrad.base`:
``fit`` learns statistics from training data only, ``transform`` applies
them — so the same fitted scaler is reused on test data without leaking
test-set statistics into it.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.typing import FeatureMatrix
from scratchgrad.utils.validation import check_array, check_is_fitted


class StandardScaler(Estimator):
    r"""Standardize features to zero mean and unit variance.

    .. math:: z = \frac{x - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are each feature's mean and
    (population, i.e. ``ddof=0``) standard deviation over the training
    data.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature mean, learned by ``fit``.
    scale_ : ndarray of shape (n_features,)
        Per-feature standard deviation, learned by ``fit``. Features with
        zero variance get ``scale_ = 1`` instead of 0, so ``transform``
        leaves them at ``x - mean_`` (i.e. 0) rather than dividing by zero.

    """

    def fit(self, X: FeatureMatrix) -> StandardScaler:
        """Learn ``mean_`` and ``scale_`` from ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data to compute per-feature mean and scale from.

        Returns
        -------
        self

        """
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Apply ``(X - mean_) / scale_`` using statistics from ``fit``."""
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Equivalent to ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Undo :meth:`transform`: ``X * scale_ + mean_``."""
        check_is_fitted(self, ["mean_", "scale_"])
        X = check_array(X)
        return X * self.scale_ + self.mean_


class MinMaxScaler(Estimator):
    r"""Scale features to a given range, ``[0, 1]`` by default.

    .. math:: x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}

    with :math:`x_{\min}`, :math:`x_{\max}` taken per-feature over the
    training data, then rescaled into ``feature_range``.

    Parameters
    ----------
    feature_range : tuple of (float, float), default=(0.0, 1.0)
        Desired output range ``(min, max)``.

    Attributes
    ----------
    data_min_ : ndarray of shape (n_features,)
        Per-feature minimum, learned by ``fit``.
    data_max_ : ndarray of shape (n_features,)
        Per-feature maximum, learned by ``fit``. Features with
        ``data_max_ == data_min_`` map to ``feature_range[0]`` rather than
        dividing by zero.

    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        """See the class docstring for the ``feature_range`` parameter."""
        self.feature_range = feature_range

    def fit(self, X: FeatureMatrix) -> MinMaxScaler:
        """Learn ``data_min_`` and ``data_max_`` from ``X``."""
        X = check_array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self

    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Map ``X`` into ``feature_range`` using statistics from ``fit``."""
        check_is_fitted(self, ["data_min_", "data_max_"])
        X = check_array(X)
        data_range = self.data_max_ - self.data_min_
        safe_range = np.where(data_range == 0.0, 1.0, data_range)
        unit_scaled = (X - self.data_min_) / safe_range  # in [0, 1], per feature
        low, high = self.feature_range
        return unit_scaled * (high - low) + low

    def fit_transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Equivalent to ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)
