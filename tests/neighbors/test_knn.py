"""Tests for scratchgrad.neighbors.KNeighborsClassifier.

Tiers (plan.md section 3): analytic hand-laid checks, an independent
brute-force reference (the correctness analogue of a gradient check — KNN
has no gradient), k-behaviour, weighting, tie-breaking, the fit/predict
contract, a behavioral check against a linear baseline, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.exceptions import NotFittedError
from scratchgrad.linear import LogisticRegression
from scratchgrad.neighbors import KNeighborsClassifier
from scratchgrad.neighbors.knn import _vote_proba
from scratchgrad.preprocessing import train_test_split


def _brute_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    metric: str,
    weights: str,
) -> np.ndarray:
    """Dead-simple O(mn) reference: one query at a time, explicit sort."""
    classes = np.unique(y_train)
    preds = []
    for xq in X_query:
        if metric == "euclidean":
            dist = np.sqrt(np.sum((X_train - xq) ** 2, axis=1))
        else:
            dist = np.sum(np.abs(X_train - xq), axis=1)
        order = np.argsort(dist, kind="stable")[:k]
        nb_labels, nb_dist = y_train[order], dist[order]
        if weights == "uniform":
            w = np.ones(k)
        elif np.any(nb_dist == 0.0):
            w = (nb_dist == 0.0).astype(float)
        else:
            w = 1.0 / nb_dist
        scores = np.array([np.sum(w[nb_labels == c]) for c in classes])
        preds.append(classes[np.argmax(scores)])
    return np.array(preds)


class TestVoteProba:
    def test_uniform_vote_is_class_frequency(self) -> None:
        labels = np.array([[0.0, 1.0, 1.0], [2.0, 2.0, 0.0]])
        weights = np.ones((2, 3))
        proba = _vote_proba(labels, weights, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(proba, [[1 / 3, 2 / 3, 0.0], [1 / 3, 0.0, 2 / 3]])

    def test_weights_scale_each_neighbor_contribution(self) -> None:
        labels = np.array([[0.0, 1.0, 1.0]])
        weights = np.array([[3.0, 1.0, 1.0]])
        proba = _vote_proba(labels, weights, np.array([0.0, 1.0]))
        np.testing.assert_allclose(proba, [[0.6, 0.4]])


class TestAnalytic:
    def test_kneighbors_returns_the_right_points_sorted(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0]])
        y = np.array([0, 0, 0, 1, 1])
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        distances, indices = model.kneighbors(np.array([[0.1, 0.1]]))
        assert set(indices[0].tolist()) == {0, 1, 2}
        assert np.all(np.diff(distances[0]) >= 0.0)
        assert distances[0, 0] == pytest.approx(np.hypot(0.1, 0.1))
        assert model.predict(np.array([[0.1, 0.1]]))[0] == 0

    def test_majority_vote_over_three_neighbors(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
        y = np.array([1, 1, 0, 0, 0])
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        # query at 1.0 -> neighbours are rows 1,0,2 (labels 1,1,0) -> vote 1
        assert model.predict(np.array([[1.0]]))[0] == 1


class TestIndependentReference:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
    @pytest.mark.parametrize("weights", ["uniform", "distance"])
    def test_matches_brute_force(self, rng, metric, weights) -> None:
        X = rng.standard_normal((80, 4))
        y = rng.integers(0, 3, size=80).astype(np.float64)
        X_query = rng.standard_normal((25, 4))

        model = KNeighborsClassifier(n_neighbors=5, metric=metric, weights=weights).fit(
            X, y
        )
        expected = _brute_predict(X, y, X_query, 5, metric, weights)
        np.testing.assert_array_equal(model.predict(X_query), expected)


class TestKBehaviour:
    def test_k1_reproduces_training_labels(self) -> None:
        X, y = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=0)
        model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_k_equals_n_predicts_global_majority(self, rng) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        y = np.array([0] * 20 + [1] * 10, dtype=np.float64)  # majority class is 0
        model = KNeighborsClassifier(n_neighbors=30).fit(X, y)
        preds = model.predict(rng.standard_normal((6, X.shape[1])))
        assert np.all(preds == 0.0)


class TestWeighting:
    def test_distance_weighting_can_flip_the_vote(self) -> None:
        X = np.array([[0.0], [10.0], [11.0]])
        y = np.array([0, 1, 1])
        query = np.array([[0.5]])
        assert (
            KNeighborsClassifier(3, weights="uniform").fit(X, y).predict(query)[0] == 1
        )
        assert (
            KNeighborsClassifier(3, weights="distance").fit(X, y).predict(query)[0] == 0
        )

    def test_query_exactly_on_a_training_point_returns_its_label(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 1, 0, 1, 0])
        model = KNeighborsClassifier(3, weights="distance").fit(X, y)
        assert model.predict(np.array([[1.0]]))[0] == 1


class TestTieBreak:
    def test_even_k_tie_breaks_to_lowest_label(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0, 1])
        model = KNeighborsClassifier(n_neighbors=4).fit(X, y)
        assert model.predict(np.array([[1.5]]))[0] == 0


class TestProba:
    def test_shape_rows_sum_to_one_and_agree_with_predict(self) -> None:
        X, y = make_blobs(n_samples=90, centers=3, cluster_std=2.0, random_state=0)
        model = KNeighborsClassifier(n_neighbors=5).fit(X, y)
        proba = model.predict_proba(X[:12])
        assert proba.shape == (12, 3)
        assert proba.sum(axis=1) == pytest.approx(np.ones(12))
        labels = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X[:12]), labels)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KNeighborsClassifier().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = KNeighborsClassifier()
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = KNeighborsClassifier(
            n_neighbors=3, metric="manhattan", weights="distance"
        )
        model.fit(X, y)
        assert model.get_params() == {
            "n_neighbors": 3,
            "metric": "manhattan",
            "weights": "distance",
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = KNeighborsClassifier().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_n_neighbors_larger_than_n_samples_raises_at_fit(self) -> None:
        X, y = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="larger than the number"):
            KNeighborsClassifier(n_neighbors=11).fit(X, y)

    def test_n_neighbors_below_one_raises(self) -> None:
        X, y = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
            KNeighborsClassifier(n_neighbors=0).fit(X, y)

    def test_unknown_metric_raises(self) -> None:
        X, y = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="metric must be one of"):
            KNeighborsClassifier(metric="cosine").fit(X, y)

    def test_unknown_weights_raises(self) -> None:
        X, y = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="weights must be one of"):
            KNeighborsClassifier(weights="gaussian").fit(X, y)

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        model = KNeighborsClassifier().fit(X, y)
        preds = model.predict(X)
        assert model.score(X, y) == pytest.approx(np.mean(preds == y))

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(KNeighborsClassifier()) == (
            "KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')"
        )


class TestBehavioral:
    def test_beats_linear_baseline_on_noisy_moons(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
        linear = LogisticRegression().fit(X_train, y_train)
        assert knn.score(X_test, y_test) > linear.score(X_test, y_test)

    def test_high_accuracy_on_well_separated_multiclass(self) -> None:
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9


class TestEdgeCases:
    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=1, centers=2, random_state=5)
        model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert model.predict(X).shape == (40,)

    def test_manhattan_metric(self, rng) -> None:
        X = rng.standard_normal((50, 3))
        y = rng.integers(0, 2, size=50).astype(np.float64)
        model = KNeighborsClassifier(n_neighbors=5, metric="manhattan").fit(X, y)
        expected = _brute_predict(X, y, X, 5, "manhattan", "uniform")
        np.testing.assert_array_equal(model.predict(X), expected)

    def test_n_neighbors_equal_to_n_samples(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=6)
        model = KNeighborsClassifier(n_neighbors=20).fit(X, y)
        assert model.predict(X).shape == (20,)
