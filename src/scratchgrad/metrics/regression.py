"""Regression metrics.

Notation follows docs/conventions.md: ``y_true``, ``y_pred`` are both
shape ``(n_samples,)``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import TargetVector


def _as_float64_pair(
    y_true: TargetVector, y_pred: TargetVector
) -> tuple[TargetVector, TargetVector]:
    """Coerce both targets to float64 ndarrays. Internal helper."""
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def mean_squared_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    r"""Mean squared error, :math:`\frac{1}{n} \sum_i (y_i - \hat{y}_i)^2`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float

    """
    y_true, y_pred = _as_float64_pair(y_true, y_pred)
    residual = y_true - y_pred
    return float(np.mean(residual**2))


def root_mean_squared_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    r"""Root mean squared error, :math:`\sqrt{\mathrm{MSE}(y, \hat{y})}`.

    In the same units as ``y``, unlike :func:`mean_squared_error`.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    r"""Mean absolute error, :math:`\frac{1}{n} \sum_i |y_i - \hat{y}_i|`.

    Less sensitive to outliers than MSE, since errors aren't squared.
    """
    y_true, y_pred = _as_float64_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: TargetVector, y_pred: TargetVector) -> float:
    r"""Coefficient of determination.

    .. math::
        R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}

    The numerator is the residual sum of squares (error the model makes);
    the denominator is the total sum of squares (error a model that always
    predicts :math:`\bar{y}`, the mean of ``y_true``, would make). A score
    of 1 means perfect prediction; 0 means "no better than predicting the
    mean"; negative means worse than that baseline.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground-truth target values.
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``y_true`` is constant, making the total sum of squares zero
        (R² is undefined — there is no variance for the model to explain).

    """
    y_true, y_pred = _as_float64_pair(y_true, y_pred)
    residual_ss = np.sum((y_true - y_pred) ** 2)  # sum_i (y_i - y_hat_i)^2
    total_ss = np.sum((y_true - np.mean(y_true)) ** 2)  # sum_i (y_i - y_bar)^2
    if total_ss == 0.0:
        raise ValueError(
            "R^2 is undefined when y_true is constant (total sum of squares is 0)."
        )
    return float(1.0 - residual_ss / total_ss)
