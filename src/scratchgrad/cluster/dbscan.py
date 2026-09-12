r"""DBSCAN: density-based clustering with noise.

Unlike K-means, there is no objective being minimised — a cluster is
*defined* combinatorially via density-reachability on the
:math:`\varepsilon`-neighbor graph (Ester, Kriegel, Sander, Xu, 1996):

- :math:`x` is a **core point** if :math:`\lvert N_\varepsilon(x) \rvert
  \ge` ``min_samples`` (counting :math:`x` itself).
- :math:`y` is directly density-reachable from a core point :math:`x` if
  :math:`y \in N_\varepsilon(x)`; density-reachability is the transitive
  closure of that relation through a chain of core points.
- A cluster is the full set of points density-reachable from one core
  point — by Ester et al.'s Lemma 2, this is independent of *which* core
  point in the cluster you start from.

**No `predict`** — a DBSCAN cluster is a property of the realized
neighbor graph of the training set, not a region of space, so there is no
principled out-of-sample rule (matches ``sklearn.cluster.DBSCAN``, which
has no ``predict`` either). ``min_samples`` and ``eps`` are the only
hyperparameters; no ``init``/``n_init``/``max_iter`` — one deterministic
pass over the graph.

Full walkthrough: ``docs/derivations/dbscan.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.pairwise import euclidean_distance, manhattan_distance
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray
from scratchgrad.utils.validation import check_array

_METRICS = {"euclidean": euclidean_distance, "manhattan": manhattan_distance}


def _core_points(
    dist: FloatArray, eps: float, min_samples: int
) -> tuple[np.ndarray, list[IntArray]]:
    """Core-point mask and each point's neighbor-index list (ascending index).

    ``min_samples`` counts a point as its own neighbor (the diagonal of
    ``dist`` is 0), matching the original paper and scikit-learn.
    """
    neighbor_mask = dist <= eps
    is_core = neighbor_mask.sum(axis=1) >= min_samples
    neighborhoods = [np.where(row)[0] for row in neighbor_mask]
    return is_core, neighborhoods


def _expand_clusters(is_core: np.ndarray, neighborhoods: list[IntArray]) -> IntArray:
    """Assign every point a cluster label (or ``-1`` for noise).

    A depth-first expansion from each unvisited core point, using an
    explicit LIFO stack rather than recursion — matches
    ``sklearn.cluster._dbscan_inner.dbscan_inner`` exactly (traversal
    order and all), which is what makes the border-point tie in
    ``docs/derivations/dbscan.md`` section 4 resolve identically to
    scikit-learn on every input, with no RNG involved on either side.
    """
    n = is_core.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    stack: list[int] = []

    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue
        i = seed
        while True:
            if labels[i] == -1:
                labels[i] = next_label
                if is_core[i]:
                    for neighbor in neighborhoods[i]:
                        if labels[neighbor] == -1:
                            stack.append(int(neighbor))
            if not stack:
                break
            i = stack.pop()
        next_label += 1

    return labels


class DBSCAN(Estimator):
    r"""Density-based clustering: core/border/noise, no ``n_clusters`` needed.

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius :math:`\varepsilon`. Two points are neighbors
        iff their distance is ``<= eps``.
    min_samples : int, default=5
        Minimum neighbor count (including the point itself) for a point to
        be a core point. Higher values require denser regions to form a
        cluster.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance used to build the neighbor graph. See
        :mod:`scratchgrad.metrics.pairwise`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,), dtype int64
        Cluster index of each training row, or ``-1`` for noise.
    core_sample_indices_ : ndarray of shape (n_core_samples,), dtype int64
        Row indices of every core point.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core point found during ``fit``.

    Notes
    -----
    Brute-force :math:`(n, n)` distance matrix — no ``algorithm`` parameter
    (no ball-tree/kd-tree), the same stance as
    :class:`scratchgrad.neighbors.KNeighborsClassifier` and
    :class:`scratchgrad.cluster.KMeans`. There is no ``predict``: a DBSCAN
    cluster is defined relative to the training set's own realized
    neighbor graph, with no principled rule for a new point (see
    ``docs/derivations/dbscan.md`` section 5). See that document for the
    full derivation, including the border-point tie-break and its exact
    scikit-learn parity.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.cluster import DBSCAN
    >>> X = np.array([[0.0, 0.0], [0.2, 0.1], [10.0, 10.0], [50.0, 50.0]])
    >>> DBSCAN(eps=1.0, min_samples=2).fit(X).labels_.tolist()
    [0, 0, -1, -1]

    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X: FeatureMatrix, y: object = None) -> DBSCAN:
        """Cluster ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to cluster.
        y : ignored
            Present for API consistency (``fit(X, y=None)``) — DBSCAN is
            unsupervised.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``eps`` is not positive, ``min_samples`` is less than 1, or
            ``metric`` is unrecognised.

        """
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}.")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}.")
        if self.metric not in _METRICS:
            raise ValueError(
                f"metric must be one of {tuple(_METRICS)}, got {self.metric!r}."
            )

        X = check_array(X)
        dist = _METRICS[self.metric](X, X)
        is_core, neighborhoods = _core_points(dist, self.eps, self.min_samples)
        self.labels_ = _expand_clusters(is_core, neighborhoods)
        self.core_sample_indices_ = np.where(is_core)[0]
        self.components_ = X[self.core_sample_indices_].copy()
        return self

    def fit_predict(self, X: FeatureMatrix, y: object = None) -> IntArray:
        """Equivalent to ``fit(X).labels_`` — fit and return the training labels."""
        return self.fit(X).labels_
