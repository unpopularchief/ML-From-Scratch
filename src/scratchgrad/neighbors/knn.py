r"""k-nearest-neighbours classification.

A lazy, non-parametric classifier: :meth:`fit` only memorises the training
data, and :meth:`predict` classifies each query point by a (optionally
distance-weighted) vote among its ``n_neighbors`` closest training points.

.. math::
    \hat{P}(c \mid x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i\,
      \mathbb{1}[y_i = c]}{\sum_{i \in \mathcal{N}_k(x)} w_i},
    \qquad w_i = \begin{cases} 1 & \text{weights="uniform"} \\
    1/d(x, x_i) & \text{weights="distance"} \end{cases}

and :meth:`predict` returns :math:`\arg\max_c \hat{P}(c \mid x)`.

Neighbours are found by brute force — the full ``(m, n)`` distance matrix
(via :mod:`scratchgrad.metrics.pairwise`) with an ``argpartition`` for the
``k`` smallest per row. There is no ``algorithm`` parameter: KD-/ball-trees
are spatial acceleration structures, not part of the method's statistics.

KNN is not scale-invariant — standardise the features first. Full
walkthrough: ``docs/derivations/knn.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.metrics.pairwise import euclidean_distance, manhattan_distance
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray, TargetVector
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y

_METRICS = {"euclidean": euclidean_distance, "manhattan": manhattan_distance}
_WEIGHTS = ("uniform", "distance")


def _vote_proba(
    labels: TargetVector, weights: FloatArray, classes: TargetVector
) -> FloatArray:
    r"""Weighted class-frequency vote over each row's neighbours.

    ``labels`` and ``weights`` are both ``(m, k)`` — the labels of, and the
    weights on, each query's ``k`` neighbours. Returns the ``(m,
    n_classes)`` matrix :math:`\hat{P}(c \mid x)` whose rows sum to 1, with
    columns ordered as ``classes`` (sorted).
    """
    proba = np.zeros((labels.shape[0], classes.shape[0]))
    for j, c in enumerate(classes):
        # Σ_i wᵢ · 𝟙[yᵢ = c]  over the k neighbours of each query row
        proba[:, j] = np.sum(weights * (labels == c), axis=1)
    return proba / proba.sum(axis=1, keepdims=True)  # each row -> a distribution


class KNeighborsClassifier(Estimator):
    r"""Classifier implementing the k-nearest-neighbours vote.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours ``k`` to vote. Must be between 1 and the
        number of training samples. Odd values avoid two-class ties.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance used to rank neighbours. See
        :mod:`scratchgrad.metrics.pairwise`.
    weights : {"uniform", "distance"}, default="uniform"
        ``"uniform"``: every neighbour votes equally. ``"distance"``: each
        neighbour's vote is weighted by ``1 / distance``, so closer points
        count for more. If a query coincides with one or more training
        points (distance 0), all weight goes to those points.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels seen in ``y``. ``predict`` returns values from
        here; ``predict_proba`` columns are in this order.

    Notes
    -----
    ``fit`` stores the training data unchanged and does no other work; all
    computation is in ``predict`` / ``predict_proba``, which build the full
    ``(n_queries, n_train)`` distance matrix. Cost is ``O(n_queries *
    n_train * n_features)`` per call. See ``docs/derivations/knn.md``.

    KNN mixes all features on their raw scales, so a wide-range feature
    dominates the distance. This estimator never rescales its inputs —
    standardise with
    :class:`scratchgrad.preprocessing.StandardScaler` before ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.neighbors import KNeighborsClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    >>> model.predict(np.array([[1.5], [11.5]])).tolist()
    [0.0, 1.0]

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    def fit(self, X: FeatureMatrix, y: TargetVector) -> KNeighborsClassifier:
        """Memorise the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``metric`` or ``weights`` is unknown, ``n_neighbors`` is
            less than 1, or ``n_neighbors`` exceeds the number of training
            samples.

        """
        if self.metric not in _METRICS:
            raise ValueError(
                f"metric must be one of {tuple(_METRICS)}, got {self.metric!r}."
            )
        if self.weights not in _WEIGHTS:
            raise ValueError(
                f"weights must be one of {_WEIGHTS}, got {self.weights!r}."
            )
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}.")

        X, y = check_X_y(X, y)
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} is larger than the number of "
                f"training samples ({X.shape[0]})."
            )
        self._X = X
        self._y = y
        self.classes_ = np.unique(y)
        return self

    def kneighbors(self, X: FeatureMatrix) -> tuple[FloatArray, IntArray]:
        """Return the distances to, and indices of, each query's ``k`` neighbours.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Query points.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distance to each neighbour, ascending within each row.
        indices : ndarray of shape (n_queries, n_neighbors)
            Row index into the training data of each neighbour, aligned
            with ``distances``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self._X.shape[1]}."
            )
        distance_matrix = _METRICS[self.metric](X, self._X)  # (n_queries, n_train)

        # The k smallest per row, unordered (O(n) partial sort), ...
        kth = self.n_neighbors - 1
        k = self.n_neighbors
        partitioned = np.argpartition(distance_matrix, kth, axis=1)[:, :k]
        part_distances = np.take_along_axis(distance_matrix, partitioned, axis=1)
        # ... then sorted by distance so ties and 1/d weighting are deterministic.
        order = np.argsort(part_distances, axis=1)
        indices = np.take_along_axis(partitioned, order, axis=1)
        distances = np.take_along_axis(part_distances, order, axis=1)
        return distances, indices

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class probabilities, shape ``(n_queries, n_classes)``.

        Column ``j`` is the (weighted) fraction of the ``k`` neighbours in
        class ``classes_[j]``. Rows sum to 1.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        distances, indices = self.kneighbors(X)
        weights = self._neighbor_weights(distances)
        return _vote_proba(self._y[indices], weights, self.classes_)

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        Ties (equal top probability) resolve to the lowest class label,
        since :func:`numpy.argmax` returns the first maximal entry and
        ``classes_`` is sorted.

        Returns
        -------
        ndarray of shape (n_queries,)
            Each entry is one of the values in ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        proba = self.predict_proba(X)  # also runs the fitted / feature-count checks
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.accuracy_score`: the fraction of
        samples whose predicted label matches the true label.
        """
        return accuracy_score(y, self.predict(X))

    def _neighbor_weights(self, distances: FloatArray) -> FloatArray:
        r"""Per-neighbour vote weights from the ``(m, k)`` distance array.

        ``"uniform"`` → all ones. ``"distance"`` → :math:`1/d`, except any
        row with a zero-distance neighbour, where all weight is placed on
        the zero-distance neighbour(s) (the query sits on a training point,
        so :math:`1/d` would be infinite).
        """
        if self.weights == "uniform":
            return np.ones_like(distances)
        with np.errstate(divide="ignore"):
            weights = 1.0 / distances  # wᵢ = 1 / d(x, xᵢ)
        exact_hit = np.isinf(weights)
        rows_with_hit = exact_hit.any(axis=1)
        # replace those rows with a 0/1 indicator of the zero-distance neighbours
        weights[rows_with_hit] = exact_hit[rows_with_hit].astype(np.float64)
        return weights
