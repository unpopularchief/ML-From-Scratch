"""Tests for scratchgrad.tree.DecisionTreeClassifier.

Tiers (plan.md section 3): analytic impurity and hand-computed split
checks; `_best_split` against an independent O(n^2) brute-force search; a
whole-tree brute-force reference builder (the correctness analogue of a
gradient check — a tree has no gradient); determinism; the regularizer
knobs; proba; feature importances; the fit/predict contract; a behavioral
check against a linear baseline; and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.exceptions import NotFittedError
from scratchgrad.linear import LogisticRegression
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier
from scratchgrad.tree.decision_tree import _best_split, _entropy, _gini


def _counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=n_classes).astype(float)


def _brute_best_split(X, y, n_classes, impurity, min_samples_leaf):
    """O(d n^2) reference: every distinct-value midpoint, impurity from scratch."""
    n_t = len(y)
    parent = float(impurity(_counts(y, n_classes)))
    best: tuple[int | None, float, float] = (None, 0.0, 0.0)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for a, b in zip(vals[:-1], vals[1:], strict=True):
            tau = 0.5 * (a + b)
            left = X[:, j] <= tau
            n_left, n_right = int(left.sum()), int((~left).sum())
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            h_left = float(impurity(_counts(y[left], n_classes)))
            h_right = float(impurity(_counts(y[~left], n_classes)))
            gain = parent - (n_left / n_t * h_left + n_right / n_t * h_right)
            if gain > best[2] + 1e-12:
                best = (j, float(tau), gain)
    return best


def _brute_predict(
    X_train,
    y_train,
    X_query,
    criterion,
    max_depth,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
):
    """A second, deliberately naive CART builder — mirrors the class's rules."""
    impurity = _gini if criterion == "gini" else _entropy
    classes = np.unique(y_train)
    y_idx = np.searchsorted(classes, y_train)
    n_classes, n_total = len(classes), len(y_train)

    def build(X, y, depth):
        value = _counts(y, n_classes) / len(y)
        imp = float(impurity(_counts(y, n_classes)))
        if (
            (max_depth is not None and depth >= max_depth)
            or len(y) < min_samples_split
            or imp == 0.0
        ):
            return ("leaf", value)
        j, tau, gain = _brute_best_split(X, y, n_classes, impurity, min_samples_leaf)
        if j is None or (len(y) / n_total) * gain < min_impurity_decrease:
            return ("leaf", value)
        left = X[:, j] <= tau
        return (
            "node",
            j,
            tau,
            build(X[left], y[left], depth + 1),
            build(X[~left], y[~left], depth + 1),
        )

    root = build(X_train, y_idx, 0)

    def descend(node, row):
        while node[0] == "node":
            _, j, tau, lft, rgt = node
            node = lft if row[j] <= tau else rgt
        return node[1]

    proba = np.array([descend(root, row) for row in X_query])
    return classes[np.argmax(proba, axis=1)]


def _leaves(node):
    return [node] if node.is_leaf else _leaves(node.left) + _leaves(node.right)


def _tree_equal(a, b):
    if a.is_leaf or b.is_leaf:
        return a.is_leaf and b.is_leaf and np.allclose(a.value, b.value)
    return (
        a.feature == b.feature
        and a.threshold == b.threshold
        and _tree_equal(a.left, b.left)
        and _tree_equal(a.right, b.right)
    )


class TestImpurity:
    def test_known_values(self) -> None:
        assert _gini(np.array([5.0, 5.0])) == pytest.approx(0.5)
        assert _gini(np.array([3.0, 0.0])) == pytest.approx(0.0)
        assert _gini(np.array([1.0, 1.0, 1.0])) == pytest.approx(2 / 3)
        assert _entropy(np.array([5.0, 5.0])) == pytest.approx(1.0)
        assert _entropy(np.array([3.0, 0.0])) == pytest.approx(0.0)
        assert _entropy(np.array([1.0, 1.0, 1.0])) == pytest.approx(np.log2(3))

    def test_row_wise_broadcasting(self) -> None:
        counts = np.array([[5.0, 5.0], [1.0, 0.0], [2.0, 2.0]])
        np.testing.assert_allclose(_gini(counts), [0.5, 0.0, 0.5])
        np.testing.assert_allclose(_entropy(counts), [1.0, 0.0, 1.0])


class TestBestSplit:
    def test_hand_computed_split(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
        y_idx = np.array([0, 0, 0, 1, 1])
        feature, threshold, gain = _best_split(X, y_idx, 2, _gini, 1)
        assert feature == 0
        assert threshold == pytest.approx(6.0)  # midpoint of 2 and 10
        assert gain == pytest.approx(0.48)  # parent gini 0.48 -> both children pure

    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    def test_matches_brute_force_search(self, rng, criterion) -> None:
        impurity = _gini if criterion == "gini" else _entropy
        X = rng.standard_normal((50, 3))
        y = rng.integers(0, 3, size=50)
        got = _best_split(X, y, 3, impurity, 3)
        expected = _brute_best_split(X, y, 3, impurity, 3)
        assert got[0] == expected[0]
        assert got[1] == pytest.approx(expected[1])
        assert got[2] == pytest.approx(expected[2])

    def test_root_split_is_the_obvious_gap(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 10.0], [0.0, 11.0]])
        y = np.array([0, 0, 0, 1, 1])  # feature 0 constant, feature 1 separates
        model = DecisionTreeClassifier().fit(X, y)
        assert model.tree_.feature == 1
        assert model.tree_.threshold == pytest.approx(6.0)


class TestBruteReference:
    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    @pytest.mark.parametrize("max_depth", [1, 3, None])
    def test_predict_matches_independent_builder(
        self, rng, criterion, max_depth
    ) -> None:
        X = rng.standard_normal((70, 3))
        y = rng.integers(0, 2, size=70).astype(np.float64)
        X_query = rng.standard_normal((30, 3))
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth).fit(
            X, y
        )
        expected = _brute_predict(X, y, X_query, criterion, max_depth)
        np.testing.assert_array_equal(model.predict(X_query), expected)


class TestDeterminism:
    def test_same_data_same_tree(self) -> None:
        X, y = make_blobs(n_samples=120, centers=3, cluster_std=2.0, random_state=0)
        a = DecisionTreeClassifier().fit(X, y)
        b = DecisionTreeClassifier().fit(X, y)
        assert _tree_equal(a.tree_, b.tree_)


class TestRegularizers:
    def test_unbounded_tree_fits_training_data_exactly(self) -> None:
        X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=0)
        model = DecisionTreeClassifier().fit(X, y)
        assert model.score(X, y) == 1.0

    def test_max_depth_one_is_a_stump(self) -> None:
        X, y = make_blobs(n_samples=100, centers=2, random_state=0)
        model = DecisionTreeClassifier(max_depth=1).fit(X, y)
        assert not model.tree_.is_leaf
        assert model.tree_.left.is_leaf and model.tree_.right.is_leaf
        assert model.max_depth_ == 1

    def test_large_min_impurity_decrease_makes_root_a_leaf(self) -> None:
        X, y = make_blobs(n_samples=100, centers=2, random_state=0)
        model = DecisionTreeClassifier(min_impurity_decrease=1.0).fit(X, y)
        assert model.tree_.is_leaf
        assert model.max_depth_ == 0
        assert np.all(model.feature_importances_ == 0.0)

    def test_min_samples_split_above_n_makes_root_a_leaf(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        model = DecisionTreeClassifier(min_samples_split=31).fit(X, y)
        assert model.tree_.is_leaf

    def test_min_samples_leaf_is_respected(self) -> None:
        X, y = make_blobs(n_samples=120, centers=3, cluster_std=2.5, random_state=1)
        model = DecisionTreeClassifier(min_samples_leaf=10).fit(X, y)
        assert all(leaf.n_samples >= 10 for leaf in _leaves(model.tree_))


class TestProba:
    def test_shape_rows_sum_to_one_and_agree_with_predict(self) -> None:
        X, y = make_blobs(n_samples=90, centers=3, cluster_std=2.0, random_state=0)
        model = DecisionTreeClassifier(max_depth=4).fit(X, y)
        proba = model.predict_proba(X[:15])
        assert proba.shape == (15, 3)
        assert proba.sum(axis=1) == pytest.approx(np.ones(15))
        labels = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X[:15]), labels)


class TestFeatureImportances:
    def test_sum_to_one_when_the_tree_splits(self) -> None:
        X, y = make_blobs(n_samples=150, centers=3, random_state=0)
        model = DecisionTreeClassifier().fit(X, y)
        assert model.feature_importances_.sum() == pytest.approx(1.0)

    def test_concentrate_on_the_informative_feature(self, rng) -> None:
        x0 = rng.standard_normal(200)
        y = (x0 > 0.0).astype(np.float64)
        X = np.column_stack([x0, rng.standard_normal(200)])  # feature 1 is noise
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)
        assert model.feature_importances_[0] > 0.9


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            DecisionTreeClassifier().predict(np.zeros((2, 2)))

    def test_predict_proba_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            DecisionTreeClassifier().predict_proba(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = DecisionTreeClassifier()
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            min_impurity_decrease=0.01,
        )
        model.fit(X, y)
        assert model.get_params() == {
            "criterion": "entropy",
            "max_depth": 3,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.01,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = DecisionTreeClassifier().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_unknown_criterion_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="criterion must be one of"):
            DecisionTreeClassifier(criterion="mse").fit(X, y)

    def test_bad_max_depth_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            DecisionTreeClassifier(max_depth=0).fit(X, y)

    def test_bad_min_samples_split_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
            DecisionTreeClassifier(min_samples_split=1).fit(X, y)

    def test_bad_min_samples_leaf_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            DecisionTreeClassifier(min_samples_leaf=0).fit(X, y)

    def test_negative_min_impurity_decrease_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="min_impurity_decrease must be >= 0"):
            DecisionTreeClassifier(min_impurity_decrease=-0.1).fit(X, y)

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        model = DecisionTreeClassifier(max_depth=3).fit(X, y)
        preds = model.predict(X)
        assert model.score(X, y) == pytest.approx(np.mean(preds == y))

    def test_repr_round_trips(self) -> None:
        assert repr(DecisionTreeClassifier()) == (
            "DecisionTreeClassifier(criterion='gini', max_depth=None, "
            "min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0)"
        )


class TestBehavioral:
    def test_beats_linear_baseline_on_moons(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        tree = DecisionTreeClassifier(max_depth=8).fit(X_train, y_train)
        linear = LogisticRegression().fit(X_train, y_train)
        assert tree.score(X_test, y_test) > linear.score(X_test, y_test)

    def test_high_accuracy_on_well_separated_multiclass(self) -> None:
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9


class TestEdgeCases:
    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=1, centers=2, random_state=5)
        model = DecisionTreeClassifier().fit(X, y)
        assert model.predict(X).shape == (40,)

    def test_single_sample(self) -> None:
        model = DecisionTreeClassifier().fit(np.array([[1.0, 2.0]]), np.array([0]))
        assert model.tree_.is_leaf
        assert model.max_depth_ == 0
        assert model.predict(np.array([[5.0, 6.0]]))[0] == 0.0

    def test_constant_features_make_a_leaf(self) -> None:
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = np.array([0, 1, 0])
        model = DecisionTreeClassifier().fit(X, y)
        assert model.tree_.is_leaf
        np.testing.assert_allclose(model.tree_.value, [2 / 3, 1 / 3])
        assert model.predict(np.array([[1.0, 1.0]]))[0] == 0.0  # argmax -> lowest label

    def test_single_class_target(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        y = np.zeros(30)
        model = DecisionTreeClassifier().fit(X, y)
        assert model.tree_.is_leaf
        assert model.tree_.impurity == 0.0
        assert np.all(model.predict(X) == 0.0)

    def test_duplicate_rows_with_conflicting_labels(self) -> None:
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([0, 1, 0, 1])  # no split can separate these
        model = DecisionTreeClassifier().fit(X, y)
        assert model.tree_.is_leaf
        np.testing.assert_allclose(model.tree_.value, [0.5, 0.5])
