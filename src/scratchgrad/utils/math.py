"""Numerically stable elementary functions used across the codebase.

These are the building blocks (not algorithms in their own right) that
show up inside logistic regression, softmax classifiers, and every neural
net loss from M3 onward. Written here once, from scratch, instead of
importing ``scipy.special`` — see plan.md section 6 for why scipy is
off-limits.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


def sigmoid(z: FloatArray) -> FloatArray:
    r"""Logistic sigmoid, :math:`\sigma(z) = 1 / (1 + e^{-z})`.

    Computed branch-wise to avoid overflow in ``exp``: for ``z >= 0`` the
    direct formula is used (``e^{-z}`` stays in ``[0, 1]``); for ``z < 0``
    the algebraically equivalent ``e^{z} / (1 + e^{z})`` is used instead,
    since ``e^{z}`` is the exponential that stays small there.

    Parameters
    ----------
    z : ndarray
        Input of any shape.

    Returns
    -------
    ndarray of the same shape as ``z``, with values in ``(0, 1)``.

    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    # z >= 0 branch: 1 / (1 + e^{-z})
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    # z < 0 branch: e^{z} / (1 + e^{z})  (equal to the formula above, but
    # keeps the exponent negative so it can't overflow)
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softmax(z: FloatArray, axis: int = -1) -> FloatArray:
    r"""Softmax, :math:`\mathrm{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}`.

    Subtracts ``max(z)`` along ``axis`` before exponentiating. This shifts
    every exponent to be ``<= 0`` (so ``exp`` cannot overflow) without
    changing the result, since the shift cancels between the numerator and
    denominator: ``e^{z_i - m} / sum_j e^{z_j - m} = e^{z_i} / sum_j e^{z_j}``.

    Parameters
    ----------
    z : ndarray
        Input logits of any shape.
    axis : int, default=-1
        Axis over which the distribution sums to 1.

    Returns
    -------
    ndarray of the same shape as ``z``, summing to 1 along ``axis``.

    """
    z = np.asarray(z, dtype=np.float64)
    shifted = z - np.max(z, axis=axis, keepdims=True)  # stability shift
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def logsumexp(z: FloatArray, axis: int = -1) -> FloatArray:
    r"""Log-sum-exp, :math:`\log \sum_j e^{z_j}`, computed stably.

    Uses the same max-subtraction trick as :func:`softmax`:
    :math:`\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}` for any
    :math:`m`, and choosing :math:`m = \max_j z_j` keeps every exponent
    :math:`\le 0`.

    Parameters
    ----------
    z : ndarray
        Input of any shape.
    axis : int, default=-1
        Axis to reduce over.

    Returns
    -------
    ndarray with ``axis`` removed.

    """
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=axis, keepdims=True)
    result = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)
