"""Tests for scratchgrad.tree.DecisionTreeRegressor.

Tiers (plan.md section 3): analytic variance / hand-computed split checks;
`_best_split` against an independent O(n^2) brute-force search (uniform and
weighted); a whole-tree brute-force reference builder (the correctness
analogue of a gradient check — a tree has no gradient); determinism; the
regularizer knobs; feature importances; the fit/predict contract; a
behavioral check against a linear baseline; and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs
from scratchgrad.exceptions import NotFittedError
from scratchgrad.linear import LinearRegression
from scratchgrad.metrics import r2_score
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeRegressor
from scratchgrad.tree.decision_tree_regressor import _best_split, _variance_from_sums


def _wvar(y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted variance of ``y``, computed directly (the brute reference)."""
    w = np.ones(len(y)) if w is None else w
    mean = np.average(y, weights=w)
    return float(np.average((y - mean) ** 2, weights=w))


def _brute_best_split(X, y, min_samples_leaf, w=None):
    """O(d n^2) reference: every distinct-value midpoint, variance from scratch."""
    n_t = len(y)
    w = np.ones(n_t) if w is None else w
    total = w.sum()
    parent = _wvar(y, w)
    best: tuple[int | None, float, float] = (None, 0.0, 0.0)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for a, b in zip(vals[:-1], vals[1:], strict=True):
            tau = 0.5 * (a + b)
            left = X[:, j] <= tau
            n_left, n_right = int(left.sum()), int((~left).sum())
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            child = (
                w[left].sum() * _wvar(y[left], w[left])
                + w[~left].sum() * _wvar(y[~left], w[~left])
            ) / total
            gain = parent - child
            if gain > best[2] + 1e-12:
                best = (j, float(tau), gain)
    return best


def _brute_predict(
    X_train,
    y_train,
    X_query,
    max_depth,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
):
    """A second, deliberately naive CART builder — mirrors the class's rules."""
    n_total = len(y_train)

    def build(X, y, depth):
        mean = float(y.mean())
        imp = float(((y - mean) ** 2).mean())
        if (
            (max_depth is not None and depth >= max_depth)
            or len(y) < min_samples_split
            or imp == 0.0
        ):
            return ("leaf", mean)
        j, tau, gain = _brute_best_split(X, y, min_samples_leaf)
        if j is None or (len(y) / n_total) * gain < min_impurity_decrease:
            return ("leaf", mean)
        left = X[:, j] <= tau
        return (
            "node",
            j,
            tau,
            build(X[left], y[left], depth + 1),
            build(X[~left], y[~left], depth + 1),
        )

    root = build(X_train, y_train, 0)

    def descend(node, row):
        while node[0] == "node":
            _, j, tau, lft, rgt = node
            node = lft if row[j] <= tau else rgt
        return node[1]

    return np.array([descend(root, row) for row in X_query])


def _leaves(node):
    return [node] if node.is_leaf else _leaves(node.left) + _leaves(node.right)


def _all_nodes(node):
    if node.is_leaf:
        return [node]
    return [node] + _all_nodes(node.left) + _all_nodes(node.right)


def _tree_equal(a, b):
    if a.is_leaf or b.is_leaf:
        return a.is_leaf and b.is_leaf and np.isclose(a.value, b.value)
    return (
        a.feature == b.feature
        and a.threshold == b.threshold
        and _tree_equal(a.left, b.left)
        and _tree_equal(a.right, b.right)
    )


def _nonlinear(rng, n):
    """y = sin(3 x0) + x1^2 + small noise — a target no line fits well."""
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
    return X, y


class TestVariance:
    def test_known_values(self) -> None:
        assert _variance_from_sums(4.0, 8.0, 16.0) == pytest.approx(0.0)  # constant 2
        # y = {0, 2}, equal weight: mean 1, variance 1
        assert _variance_from_sums(2.0, 2.0, 4.0) == pytest.approx(1.0)
        # y = {1, 5} with weights 3, 1: mean 2, variance 3
        assert _variance_from_sums(4.0, 8.0, 28.0) == pytest.approx(3.0)

    def test_clamped_at_zero(self) -> None:
        # a tiny negative from cancellation is clamped, not propagated
        assert _variance_from_sums(1.0, 1.0, 1.0 - 1e-18) == 0.0

    def test_row_wise_broadcasting(self) -> None:
        w = np.array([2.0, 2.0, 4.0])
        s = np.array([2.0, 0.0, 8.0])
        q = np.array([4.0, 0.0, 28.0])
        np.testing.assert_allclose(_variance_from_sums(w, s, q), [1.0, 0.0, 3.0])

    def test_node_mean_minimises_sse(self, rng) -> None:
        y = rng.standard_normal(50)
        grid = np.linspace(y.min(), y.max(), 400)
        sse = [np.sum((y - c) ** 2) for c in grid]
        assert grid[int(np.argmin(sse))] == pytest.approx(y.mean(), abs=0.05)


class TestBestSplit:
    def test_hand_computed_split(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
        y = np.array([1.0, 1.0, 1.0, 9.0, 9.0])
        feature, threshold, gain = _best_split(X, y, 1)
        assert feature == 0
        assert threshold == pytest.approx(6.0)  # midpoint of 2 and 10
        # parent variance 12.96 -> both children constant, so ΔH = parent
        assert gain == pytest.approx(_wvar(y))

    @pytest.mark.parametrize("weighted", [False, True])
    def test_matches_brute_force_search(self, rng, weighted) -> None:
        X = rng.standard_normal((60, 3))
        y = X[:, 0] ** 2 - X[:, 1] + rng.standard_normal(60)
        w = rng.uniform(0.2, 3.0, size=60) if weighted else None
        got = _best_split(X, y, 3, sample_weight=w)
        expected = _brute_best_split(X, y, 3, w)
        assert got[0] == expected[0]
        assert got[1] == pytest.approx(expected[1])
        assert got[2] == pytest.approx(expected[2])

    def test_root_split_is_the_obvious_step(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 10.0], [0.0, 11.0]])
        y = np.array([0.0, 0.0, 0.0, 5.0, 5.0])  # feature 0 constant, feature 1 steps
        model = DecisionTreeRegressor().fit(X, y)
        assert model.tree_.feature == 1
        assert model.tree_.threshold == pytest.approx(6.0)


class TestBruteReference:
    @pytest.mark.parametrize("max_depth", [1, 3, None])
    def test_predict_matches_independent_builder(self, rng, max_depth) -> None:
        X, y = _nonlinear(rng, 80)
        X_query = rng.uniform(-2.0, 2.0, size=(30, 2))
        model = DecisionTreeRegressor(max_depth=max_depth).fit(X, y)
        expected = _brute_predict(X, y, X_query, max_depth)
        np.testing.assert_allclose(model.predict(X_query), expected)


class TestDeterminism:
    def test_same_data_same_tree(self, rng) -> None:
        X, y = _nonlinear(rng, 120)
        a = DecisionTreeRegressor(max_depth=5).fit(X, y)
        b = DecisionTreeRegressor(max_depth=5).fit(X, y)
        assert _tree_equal(a.tree_, b.tree_)

    def test_seeded_run_is_reproducible(self, rng) -> None:
        X = rng.standard_normal((200, 6))
        y = X @ rng.standard_normal(6) + rng.standard_normal(200)
        a = DecisionTreeRegressor(max_features="sqrt", random_state=7).fit(X, y)
        b = DecisionTreeRegressor(max_features="sqrt", random_state=7).fit(X, y)
        assert _tree_equal(a.tree_, b.tree_)

    def test_different_seed_generally_differs(self, rng) -> None:
        X = rng.standard_normal((200, 6))
        y = X @ rng.standard_normal(6) + rng.standard_normal(200)
        a = DecisionTreeRegressor(max_features=2, random_state=1).fit(X, y)
        b = DecisionTreeRegressor(max_features=2, random_state=2).fit(X, y)
        assert not _tree_equal(a.tree_, b.tree_)

    def test_none_with_a_seed_matches_the_deterministic_tree(self, rng) -> None:
        X, y = _nonlinear(rng, 150)
        plain = DecisionTreeRegressor(max_depth=5).fit(X, y)
        seeded = DecisionTreeRegressor(max_depth=5, random_state=123).fit(X, y)
        assert _tree_equal(plain.tree_, seeded.tree_)


class TestFeatureSubsampling:
    def test_resolved_attribute_is_exposed(self, rng) -> None:
        X = rng.standard_normal((40, 9))
        y = rng.standard_normal(40)
        model = DecisionTreeRegressor(max_features="sqrt").fit(X, y)
        assert model.max_features_ == 3

    def test_every_realised_split_is_on_a_present_feature(self, rng) -> None:
        X = rng.standard_normal((200, 8))
        y = X[:, 0] ** 2 + X[:, 3] + rng.standard_normal(200)
        model = DecisionTreeRegressor(max_features=3, random_state=0).fit(X, y)
        internal = [n for n in _all_nodes(model.tree_) if not n.is_leaf]
        assert internal
        assert all(0 <= n.feature < 8 for n in internal)

    def test_bad_max_features_raises(self, rng) -> None:
        X = rng.standard_normal((30, 10))
        y = rng.standard_normal(30)
        with pytest.raises(ValueError, match="max_features"):
            DecisionTreeRegressor(max_features="auto").fit(X, y)

    def test_max_features_one_still_fits_a_step(self, rng) -> None:
        X = rng.uniform(0.0, 10.0, size=(200, 4))
        y = (X[:, 0] > 5.0).astype(np.float64)
        model = DecisionTreeRegressor(max_depth=8, max_features=1, random_state=0)
        model.fit(X, y)
        assert model.score(X, y) > 0.95


class TestSampleWeight:
    def test_none_equals_explicit_ones(self, rng) -> None:
        X, y = _nonlinear(rng, 120)
        a = DecisionTreeRegressor(max_depth=4).fit(X, y)
        b = DecisionTreeRegressor(max_depth=4).fit(X, y, sample_weight=np.ones(120))
        assert _tree_equal(a.tree_, b.tree_)

    def test_integer_weights_equal_row_duplication(self, rng) -> None:
        X, y = _nonlinear(rng, 70)
        w = rng.integers(1, 4, size=70)
        X_query = rng.uniform(-2.0, 2.0, size=(30, 2))

        weighted = DecisionTreeRegressor(max_depth=4).fit(X, y, sample_weight=w)
        duplicated = DecisionTreeRegressor(max_depth=4).fit(
            np.repeat(X, w, axis=0), np.repeat(y, w)
        )

        assert _tree_equal(weighted.tree_, duplicated.tree_)
        np.testing.assert_allclose(
            weighted.predict(X_query), duplicated.predict(X_query)
        )
        np.testing.assert_allclose(
            weighted.feature_importances_, duplicated.feature_importances_
        )

    def test_zero_weight_rows_are_dropped_outright(self, rng) -> None:
        X, y = _nonlinear(rng, 100)
        keep = rng.random(100) > 0.3
        w = keep.astype(np.float64)

        with_zeros = DecisionTreeRegressor(max_depth=5).fit(X, y, sample_weight=w)
        subset = DecisionTreeRegressor(max_depth=5).fit(X[keep], y[keep])
        assert _tree_equal(with_zeros.tree_, subset.tree_)

    def test_leaf_value_is_the_weighted_mean(self) -> None:
        # Two identical rows with conflicting targets -> the root stays a
        # leaf whose value is the weight-normalised mean, not the row mean.
        X = np.array([[0.0], [0.0]])
        y = np.array([0.0, 4.0])
        model = DecisionTreeRegressor().fit(X, y, sample_weight=np.array([3.0, 1.0]))
        assert model.tree_.is_leaf
        assert model.tree_.value == pytest.approx(1.0)
        assert model.predict(np.array([[0.0]]))[0] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "bad",
        [np.ones(5), np.full(20, -1.0), np.zeros(20), np.array([np.nan] * 20)],
    )
    def test_bad_sample_weight_raises(self, bad) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="sample_weight"):
            DecisionTreeRegressor().fit(X, y.astype(np.float64), sample_weight=bad)


class TestTargetScaling:
    # A variance decrease has the units of y^2, so a gain floor that is a bare
    # constant would reject real splits once y is small enough. The split
    # search must be invariant to the units of the target.
    @pytest.mark.parametrize("scale", [1e6, 1e3, 1.0, 1e-3, 1e-6, 1e-9])
    def test_tree_is_invariant_to_target_scale(self, rng, scale) -> None:
        X = rng.uniform(-3.0, 3.0, size=(120, 3))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2

        base = DecisionTreeRegressor(max_depth=6).fit(X, y)
        scaled = DecisionTreeRegressor(max_depth=6).fit(X, y * scale)

        assert len(_leaves(scaled.tree_)) == len(_leaves(base.tree_))
        assert scaled.max_depth_ == base.max_depth_
        # same partition, and every prediction is just the base one rescaled
        np.testing.assert_allclose(
            scaled.predict(X), base.predict(X) * scale, rtol=1e-9
        )

    @pytest.mark.parametrize("offset", [1e3, 1e6, 1e9])
    def test_tree_is_invariant_to_target_offset(self, rng, offset) -> None:
        # Variance is shift-invariant, so y + c must give the same tree. The
        # sum form q/w - (s/w)^2 only survives this because _best_split
        # centres y first; uncentred, offset 1e9 makes the computed variance
        # collapse to 0.0 and every node look pure.
        X = rng.uniform(-3.0, 3.0, size=(200, 2))
        y = np.sin(X[:, 0])

        base = DecisionTreeRegressor(max_depth=5).fit(X, y)
        shifted = DecisionTreeRegressor(max_depth=5).fit(X, y + offset)

        assert len(_leaves(shifted.tree_)) == len(_leaves(base.tree_))
        np.testing.assert_allclose(
            shifted.predict(X) - offset, base.predict(X), atol=1e-6
        )

    @pytest.mark.parametrize("scale", [1e-6, 1e-9])
    def test_small_targets_still_split(self, rng, scale) -> None:
        # Regression guard: an absolute 1e-12 gain floor silently collapsed
        # this to a single leaf with R^2 == 0.
        X = rng.uniform(-3.0, 3.0, size=(120, 3))
        y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2) * scale

        model = DecisionTreeRegressor(max_depth=6).fit(X, y)

        assert not model.tree_.is_leaf, "root collapsed to a leaf"
        assert model.score(X, y) > 0.9


class TestRegularizers:
    def test_unbounded_tree_fits_a_clean_step_exactly(self, rng) -> None:
        X = rng.uniform(0.0, 10.0, size=(300, 1))
        y = np.where(X[:, 0] < 3.0, 0.0, np.where(X[:, 0] < 7.0, 5.0, 2.0))
        model = DecisionTreeRegressor().fit(X, y)
        assert model.score(X, y) > 0.999

    def test_max_depth_one_is_a_stump(self, rng) -> None:
        X, y = _nonlinear(rng, 100)
        model = DecisionTreeRegressor(max_depth=1).fit(X, y)
        assert not model.tree_.is_leaf
        assert model.tree_.left.is_leaf and model.tree_.right.is_leaf
        assert model.max_depth_ == 1

    def test_large_min_impurity_decrease_makes_root_a_leaf(self, rng) -> None:
        X, y = _nonlinear(rng, 100)
        model = DecisionTreeRegressor(min_impurity_decrease=100.0).fit(X, y)
        assert model.tree_.is_leaf
        assert model.max_depth_ == 0
        assert np.all(model.feature_importances_ == 0.0)

    def test_min_samples_split_above_n_makes_root_a_leaf(self, rng) -> None:
        X, y = _nonlinear(rng, 30)
        model = DecisionTreeRegressor(min_samples_split=31).fit(X, y)
        assert model.tree_.is_leaf

    def test_min_samples_leaf_is_respected(self, rng) -> None:
        X, y = _nonlinear(rng, 150)
        model = DecisionTreeRegressor(min_samples_leaf=10).fit(X, y)
        assert all(leaf.n_samples >= 10 for leaf in _leaves(model.tree_))


class TestFeatureImportances:
    def test_sum_to_one_when_the_tree_splits(self, rng) -> None:
        X, y = _nonlinear(rng, 150)
        model = DecisionTreeRegressor(max_depth=5).fit(X, y)
        assert model.feature_importances_.sum() == pytest.approx(1.0)

    def test_concentrate_on_the_informative_feature(self, rng) -> None:
        x0 = rng.uniform(-2.0, 2.0, size=200)
        y = np.where(x0 > 0.0, 3.0, -3.0)
        X = np.column_stack([x0, rng.standard_normal(200)])  # feature 1 is noise
        model = DecisionTreeRegressor(max_depth=3).fit(X, y)
        assert model.feature_importances_[0] > 0.9


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            DecisionTreeRegressor().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self, rng) -> None:
        X, y = _nonlinear(rng, 40)
        model = DecisionTreeRegressor()
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self, rng) -> None:
        X, y = _nonlinear(rng, 40)
        model = DecisionTreeRegressor(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            min_impurity_decrease=0.01,
        )
        model.fit(X, y)
        assert model.get_params() == {
            "criterion": "squared_error",
            "max_depth": 3,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "min_impurity_decrease": 0.01,
            "max_features": None,
            "random_state": None,
        }

    def test_predict_rejects_wrong_feature_count(self, rng) -> None:
        X = rng.standard_normal((40, 3))
        y = rng.standard_normal(40)
        model = DecisionTreeRegressor().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_predict_output_shape(self, rng) -> None:
        X, y = _nonlinear(rng, 60)
        model = DecisionTreeRegressor(max_depth=3).fit(X, y)
        assert model.predict(X[:12]).shape == (12,)

    def test_unknown_criterion_raises(self, rng) -> None:
        X, y = _nonlinear(rng, 20)
        with pytest.raises(ValueError, match="criterion must be one of"):
            DecisionTreeRegressor(criterion="absolute_error").fit(X, y)

    def test_bad_max_depth_raises(self, rng) -> None:
        X, y = _nonlinear(rng, 20)
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            DecisionTreeRegressor(max_depth=0).fit(X, y)

    def test_bad_min_samples_split_raises(self, rng) -> None:
        X, y = _nonlinear(rng, 20)
        with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
            DecisionTreeRegressor(min_samples_split=1).fit(X, y)

    def test_bad_min_samples_leaf_raises(self, rng) -> None:
        X, y = _nonlinear(rng, 20)
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            DecisionTreeRegressor(min_samples_leaf=0).fit(X, y)

    def test_negative_min_impurity_decrease_raises(self, rng) -> None:
        X, y = _nonlinear(rng, 20)
        with pytest.raises(ValueError, match="min_impurity_decrease must be >= 0"):
            DecisionTreeRegressor(min_impurity_decrease=-0.1).fit(X, y)

    def test_score_is_r2(self, rng) -> None:
        X, y = _nonlinear(rng, 60)
        model = DecisionTreeRegressor(max_depth=3).fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))

    def test_repr_round_trips(self) -> None:
        assert repr(DecisionTreeRegressor()) == (
            "DecisionTreeRegressor(criterion='squared_error', max_depth=None, "
            "min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, "
            "max_features=None, random_state=None)"
        )


class TestBehavioral:
    def test_beats_linear_baseline_on_a_nonlinear_target(self, rng) -> None:
        X, y = _nonlinear(rng, 500)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        tree = DecisionTreeRegressor(max_depth=6).fit(X_train, y_train)
        linear = LinearRegression().fit(X_train, y_train)
        assert tree.score(X_test, y_test) > linear.score(X_test, y_test)

    def test_recovers_a_one_dimensional_step_function(self, rng) -> None:
        X = rng.uniform(0.0, 10.0, size=(400, 1))
        y = np.where(X[:, 0] < 4.0, -2.0, 3.0)
        X_test = np.linspace(0.1, 9.9, 200).reshape(-1, 1)
        y_test = np.where(X_test[:, 0] < 4.0, -2.0, 3.0)
        model = DecisionTreeRegressor(max_depth=4).fit(X, y)
        np.testing.assert_allclose(model.predict(X_test), y_test)


class TestEdgeCases:
    def test_single_feature(self, rng) -> None:
        X = rng.standard_normal((40, 1))
        y = X[:, 0] + rng.standard_normal(40)
        model = DecisionTreeRegressor().fit(X, y)
        assert model.predict(X).shape == (40,)

    def test_single_sample(self) -> None:
        model = DecisionTreeRegressor().fit(np.array([[1.0, 2.0]]), np.array([7.0]))
        assert model.tree_.is_leaf
        assert model.max_depth_ == 0
        assert model.predict(np.array([[5.0, 6.0]]))[0] == pytest.approx(7.0)

    def test_constant_target(self, rng) -> None:
        X = rng.standard_normal((30, 2))
        y = np.full(30, 3.5)
        model = DecisionTreeRegressor().fit(X, y)
        assert model.tree_.is_leaf
        assert model.tree_.impurity == 0.0
        np.testing.assert_allclose(model.predict(X), 3.5)

    def test_constant_features_make_a_leaf(self) -> None:
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        y = np.array([0.0, 2.0, 4.0])
        model = DecisionTreeRegressor().fit(X, y)
        assert model.tree_.is_leaf
        assert model.tree_.value == pytest.approx(2.0)

    def test_duplicate_rows_with_conflicting_targets(self) -> None:
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([0.0, 2.0, 10.0, 12.0])  # no split separates the pairs
        model = DecisionTreeRegressor(max_depth=1).fit(X, y)
        # the one useful split is at x = 0.5; each leaf is its pair's mean
        assert model.tree_.threshold == pytest.approx(0.5)
        np.testing.assert_allclose(
            sorted(leaf.value for leaf in _leaves(model.tree_)), [1.0, 11.0]
        )
