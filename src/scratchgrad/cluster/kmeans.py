r"""K-means clustering.

Partitions ``X`` into ``n_clusters`` groups by alternating two exact
argmins (Lloyd's algorithm) on the within-cluster sum of squares:

.. math::
    J(\{\mu_k\}, \{c_i\}) = \sum_i \lVert x_i - \mu_{c_i} \rVert^2

**Assignment step**: :math:`c_i \leftarrow \arg\min_k \lVert x_i - \mu_k
\rVert^2` (nearest centroid). **Update step**: :math:`\mu_k \leftarrow
\mathrm{mean}(\{x_i : c_i = k\})` (the SSE-minimising constant per
cluster). Each step is the exact minimiser of its subproblem holding the
other block fixed, so :math:`J` is non-increasing across iterations and
the algorithm reaches a local optimum in finitely many steps — never
guaranteed global, which is why ``init`` and ``n_init`` matter.

Full walkthrough: ``docs/derivations/kmeans.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.pairwise import euclidean_distance
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)

_INIT_STRATEGIES = ("k-means++", "random")


def _init_random(
    X: FeatureMatrix, n_clusters: int, rng: np.random.Generator
) -> FloatArray:
    """Pick ``n_clusters`` distinct rows of ``X`` uniformly at random."""
    indices = rng.choice(X.shape[0], size=n_clusters, replace=False)
    return X[indices].copy()


def _init_kmeans_plusplus(
    X: FeatureMatrix, n_clusters: int, rng: np.random.Generator
) -> FloatArray:
    """k-means++ seeding: each new centroid drawn with probability ∝ D(x)^2.

    ``D(x)`` is a point's distance to the nearest centroid already chosen
    — unrepresented regions of the data are far more likely to seed the
    next centroid. See ``docs/derivations/kmeans.md`` §5.
    """
    n_samples = X.shape[0]
    centers = np.empty((n_clusters, X.shape[1]))
    centers[0] = X[rng.integers(n_samples)]
    # D(x)^2 to the nearest of the centers chosen so far.
    closest_sq_dist = euclidean_distance(X, centers[:1]).ravel() ** 2
    for k in range(1, n_clusters):
        total = closest_sq_dist.sum()
        # Degenerate fallback (every point ties an already-chosen
        # centroid, e.g. all-duplicate rows): sample uniformly instead of
        # dividing by zero.
        probs = (
            np.full(n_samples, 1.0 / n_samples)
            if total == 0.0
            else closest_sq_dist / total
        )
        next_index = rng.choice(n_samples, p=probs)
        centers[k] = X[next_index]
        new_sq_dist = euclidean_distance(X, centers[k : k + 1]).ravel() ** 2
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)  # D(x) shrinks
    return centers


def _assign(X: FeatureMatrix, centers: FloatArray) -> tuple[IntArray, FloatArray]:
    """Nearest-centroid assignment — the exact argmin of §3 step 1.

    Returns the ``(n,)`` label of each row's nearest centroid and the full
    ``(n, n_clusters)`` distance matrix, which the caller reuses for the
    update step and the inertia rather than recomputing it.
    """
    dist = euclidean_distance(X, centers)  # (n, n_clusters)
    labels = np.argmin(dist, axis=1)
    return labels, dist


def _update_centroids(
    X: FeatureMatrix, labels: IntArray, dist: FloatArray, n_clusters: int
) -> FloatArray:
    """Per-cluster mean — the exact argmin of §3 step 2.

    A cluster left with zero members has its centroid relocated to
    whichever assigned point is farthest (in squared distance) from its
    own cluster's centroid, "stealing" it into a new singleton cluster —
    see ``docs/derivations/kmeans.md`` §7. Each empty cluster steals a
    distinct point, and a stolen point is excluded from its *original*
    cluster's mean (two passes below: relocate first, then average) —
    otherwise that point would count toward two centroids at once.
    """
    labels = labels.copy()
    counts = np.bincount(labels, minlength=n_clusters)
    own_sq_dist = dist[np.arange(dist.shape[0]), labels] ** 2
    stolen = np.zeros(dist.shape[0], dtype=bool)
    new_centers = np.empty((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        if counts[k] == 0:
            candidates = np.where(stolen, -np.inf, own_sq_dist)
            far_index = np.argmax(candidates)
            new_centers[k] = X[far_index]
            stolen[far_index] = True
            labels[far_index] = k  # remove it from its original cluster's mean below
    for k in range(n_clusters):
        if counts[k] > 0:
            new_centers[k] = X[labels == k].mean(axis=0)
    return new_centers


def _inertia(dist: FloatArray, labels: IntArray) -> float:
    """Sum of each point's squared distance to its assigned centroid."""
    return float(np.sum(dist[np.arange(dist.shape[0]), labels] ** 2))


def _lloyd(
    X: FeatureMatrix,
    centers: FloatArray,
    n_clusters: int,
    max_iter: int,
    tol: float,
) -> tuple[FloatArray, IntArray, float, int]:
    """Run Lloyd's algorithm to convergence (or ``max_iter``) from ``centers``.

    Returns the final ``(centers, labels, inertia, n_iter)`` — see
    ``docs/derivations/kmeans.md`` §4/§6 for the monotonicity argument and
    the stopping rule.
    """
    for _iteration in range(1, max_iter + 1):
        labels, dist = _assign(X, centers)
        new_centers = _update_centroids(X, labels, dist, n_clusters)
        shift = float(np.sum((new_centers - centers) ** 2))  # Σ_k ||μ_new - μ_old||^2
        centers = new_centers
        if shift <= tol:
            break
    labels, dist = _assign(X, centers)  # re-assign against the final centers
    return centers, labels, _inertia(dist, labels), _iteration


class KMeans(Estimator):
    r"""Partition data into ``n_clusters`` groups by within-cluster SSE.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters ``K``. Must be between 1 and ``n_samples``.
    init : {"k-means++", "random"} or ndarray of shape (n_clusters, n_features)
        Seeding strategy, default ``"k-means++"``: seeds spread out via a
        D(x)^2-weighted draw (§5). ``"random"``: ``n_clusters`` distinct
        training points chosen uniformly. An explicit ``(n_clusters,
        n_features)`` array of starting centroids skips both —
        deterministic, and forces ``n_init`` to 1 (repeating an identical
        deterministic run would be pointless).
    n_init : int, default=10
        Number of independent random restarts; the run with the lowest
        final ``inertia_`` is kept. Ignored (fixed to 1) when ``init`` is
        an explicit array.
    max_iter : int, default=300
        Maximum Lloyd iterations per run.
    tol : float, default=1e-4
        Relative convergence tolerance: a run stops early once the summed
        squared centroid shift falls to ``tol`` times the data's mean
        per-feature variance. See §6.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds ``"k-means++"``/``"random"`` initialisation. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centroids of the kept (lowest-inertia) run.
    labels_ : ndarray of shape (n_samples,), dtype int64
        Cluster index (into ``cluster_centers_``) assigned to each
        training row.
    inertia_ : float
        Within-cluster sum of squares of the kept run — the objective
        :math:`J` from §2, evaluated at the final assignment/centroids.
    n_iter_ : int
        Number of Lloyd iterations run by the kept run.

    Notes
    -----
    Euclidean distance only — the update step's mean-minimises-SSE
    argument (§3) is specific to squared L2. Like KNN, this estimator
    never rescales its inputs; standardise with
    :class:`scratchgrad.preprocessing.StandardScaler` first if features
    are on very different scales. See ``docs/derivations/kmeans.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.cluster import KMeans
    >>> X = np.array([[0.0, 0.0], [0.2, -0.1], [10.0, 10.0], [10.1, 9.9]])
    >>> model = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> model.predict(np.array([[0.1, 0.1], [9.9, 10.0]])).tolist()
    [1, 0]

    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | FloatArray = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: FeatureMatrix, y: object = None) -> KMeans:
        """Cluster ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to cluster.
        y : ignored
            Present for API consistency (``fit(X, y=None)``) — K-means is
            unsupervised.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_clusters``, ``n_init``, or ``max_iter`` is not
            positive, ``tol`` is negative, ``n_clusters`` exceeds the
            number of samples, ``init`` is an unrecognised string, or an
            array ``init`` has the wrong shape.

        """
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}.")
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")

        X = check_array(X)
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} is larger than the number of "
                f"samples ({X.shape[0]})."
            )

        explicit_init = isinstance(self.init, np.ndarray)
        if explicit_init:
            init_array = np.asarray(self.init, dtype=np.float64)
            expected_shape = (self.n_clusters, X.shape[1])
            if init_array.shape != expected_shape:
                raise ValueError(
                    f"init array must have shape {expected_shape}, got "
                    f"{init_array.shape}."
                )
        elif self.init not in _INIT_STRATEGIES:
            raise ValueError(
                f"init must be one of {_INIT_STRATEGIES} or an ndarray, got "
                f"{self.init!r}."
            )

        rng = check_random_state(self.random_state)
        # tol is scaled by the data's own spread (§6) so one absolute
        # number is meaningful across differently-scaled datasets.
        tol_scaled = self.tol * X.var(axis=0).mean()
        n_init = 1 if explicit_init else self.n_init

        best: tuple[FloatArray, IntArray, float, int] | None = None
        for _ in range(n_init):
            if explicit_init:
                centers = init_array.copy()
            elif self.init == "k-means++":
                centers = _init_kmeans_plusplus(X, self.n_clusters, rng)
            else:
                centers = _init_random(X, self.n_clusters, rng)

            result = _lloyd(X, centers, self.n_clusters, self.max_iter, tol_scaled)
            if best is None or result[2] < best[2]:
                best = result

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: FeatureMatrix) -> IntArray:
        """Assign each row of ``X`` to its nearest fitted centroid.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X)
        self._check_n_features(X)
        labels, _ = _assign(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X: FeatureMatrix, y: object = None) -> IntArray:
        """Equivalent to ``fit(X).labels_`` — fit and return the training labels."""
        return self.fit(X).labels_

    def transform(self, X: FeatureMatrix) -> FloatArray:
        """Return the distance from each row of ``X`` to every centroid.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "cluster_centers_")
        X = check_array(X)
        self._check_n_features(X)
        return euclidean_distance(X, self.cluster_centers_)

    def fit_transform(self, X: FeatureMatrix, y: object = None) -> FloatArray:
        """Equivalent to ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)

    def score(self, X: FeatureMatrix, y: object = None) -> float:
        """Return the negative inertia of ``X`` under the fitted centroids.

        Negative, following scikit-learn's convention that ``score`` is
        "higher is better" while inertia itself is "lower is better".
        """
        dist = self.transform(X)
        labels = np.argmin(dist, axis=1)
        return -_inertia(dist, labels)

    def _check_n_features(self, X: FeatureMatrix) -> None:
        """Raise ``ValueError`` if ``X`` doesn't match the fitted feature count."""
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.cluster_centers_.shape[1]}."
            )
