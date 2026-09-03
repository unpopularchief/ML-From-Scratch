"""Pairwise distance and similarity functions.

Used by KNN (M1), KMeans/DBSCAN (M2), and anywhere else "how far apart are
these two points" needs a name. Each function takes two matrices of shape
``(n_samples_a, n_features)`` and ``(n_samples_b, n_features)`` and returns
the full ``(n_samples_a, n_samples_b)`` matrix of pairwise values —
callers pass single points as a matrix with one row.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FeatureMatrix, FloatArray


def euclidean_distance(a: FeatureMatrix, b: FeatureMatrix) -> FloatArray:
    r"""Pairwise Euclidean (L2) distance.

    .. math:: d(x, y) = \sqrt{\sum_k (x_k - y_k)^2}

    Computed via the expansion :math:`\|x - y\|^2 = \|x\|^2 - 2 x \cdot y +
    \|y\|^2` so the work is one matrix multiplication (:math:`a b^T`) plus
    two vectors of squared norms, rather than materializing every pairwise
    difference explicitly.

    Parameters
    ----------
    a : ndarray of shape (n_samples_a, n_features)
        First set of points.
    b : ndarray of shape (n_samples_b, n_features)
        Second set of points.

    Returns
    -------
    ndarray of shape (n_samples_a, n_samples_b)

    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_sq = np.sum(a**2, axis=1, keepdims=True)  # ||x||^2, shape (n_a, 1)
    b_sq = np.sum(b**2, axis=1, keepdims=True)  # ||y||^2, shape (n_b, 1)
    cross = a @ b.T  # x . y, shape (n_a, n_b)
    # Squared distance can be a tiny negative number due to floating-point
    # cancellation when x == y; clip to 0 before the square root.
    sq_dist = np.maximum(a_sq - 2.0 * cross + b_sq.T, 0.0)
    return np.sqrt(sq_dist)


def manhattan_distance(a: FeatureMatrix, b: FeatureMatrix) -> FloatArray:
    r"""Pairwise Manhattan (L1) distance, :math:`d(x, y) = \sum_k |x_k - y_k|`.

    Computed with an explicit broadcast difference (unlike
    :func:`euclidean_distance`'s matrix-multiply trick, which relies on a
    square that L1 doesn't have), so memory use is ``O(n_a * n_b *
    n_features)`` rather than ``O(n_a * n_b)``.

    Parameters
    ----------
    a : ndarray of shape (n_samples_a, n_features)
        First set of points.
    b : ndarray of shape (n_samples_b, n_features)
        Second set of points.

    Returns
    -------
    ndarray of shape (n_samples_a, n_samples_b)

    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # shape (n_a, n_b, n_features)
    return np.sum(np.abs(diff), axis=2)


def cosine_similarity(a: FeatureMatrix, b: FeatureMatrix) -> FloatArray:
    r"""Pairwise cosine similarity, :math:`\frac{x \cdot y}{\|x\| \|y\|}`.

    The cosine of the angle between ``x`` and ``y``: 1 means identical
    direction, 0 means orthogonal, -1 means opposite direction —
    independent of each vector's magnitude, unlike the distances above.

    Parameters
    ----------
    a : ndarray of shape (n_samples_a, n_features)
        First set of points.
    b : ndarray of shape (n_samples_b, n_features)
        Second set of points.

    Returns
    -------
    ndarray of shape (n_samples_a, n_samples_b)

    Raises
    ------
    ValueError
        If any row of ``a`` or ``b`` is the zero vector (norm 0, making
        the similarity undefined).

    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)  # ||x||, shape (n_a, 1)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)  # ||y||, shape (n_b, 1)
    if np.any(a_norm == 0.0) or np.any(b_norm == 0.0):
        raise ValueError("cosine_similarity is undefined for a zero vector.")
    cross = a @ b.T  # x . y, shape (n_a, n_b)
    return cross / (a_norm @ b_norm.T)
