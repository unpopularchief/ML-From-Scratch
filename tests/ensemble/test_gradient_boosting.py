"""Tests for scratchgrad.ensemble.GradientBoostingClassifier.

Tiers (plan.md section 3): the ensemble has no single differentiable
objective to gradient-check, so the correctness analogues are (a) the
module-level gradient / Hessian / init / Newton-leaf helpers checked
against hand math, (b) one boosting round with `init="zero"`,
`learning_rate=1` recomputed independently (leaf values, raw scores, and
the residuals handed to round 2), and (c) `sample_weight` integer weights
matching a fit on the row-duplicated data. Plus the training-loss curve,
boosting behaviour, shrinkage, `subsample`, `init`, the staged iterators,
`predict_proba` / `decision_function`, the fit/predict contract, and edges.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import GradientBoostingClassifier
from scratchgrad.ensemble.gradient_boosting import (
    _binary_negative_gradient,
    _log_odds,
    _multinomial_negative_gradient,
    _newton_leaf_value,
    _prior_logits,
    _pseudo_hessian,
)
from scratchgrad.exceptions import NotFittedError
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scratchgrad.tree.decision_tree import _leaf_for
from scratchgrad.utils.math import sigmoid, softmax


def _leaf_groups(tree, X):
    """Map each leaf node of a fitted tree to the row indices that reach it."""
    nodes: dict[int, object] = {}
    rows: dict[int, list[int]] = defaultdict(list)
    for i, sample in enumerate(X):
        leaf = _leaf_for(tree.tree_, sample)
        nodes[id(leaf)] = leaf
        rows[id(leaf)].append(i)
    return [(nodes[key], np.array(idx)) for key, idx in rows.items()]


class TestHelpers:
    def test_log_odds_hand_value(self) -> None:
        y01 = np.array([1.0, 1.0, 1.0, 0.0])  # base rate 3/4
        w = np.ones(4)
        assert _log_odds(y01, w) == pytest.approx(np.log(0.75 / 0.25))

    def test_log_odds_is_weighted(self) -> None:
        y01 = np.array([1.0, 0.0])
        w = np.array([3.0, 1.0])  # weighted base rate 3/4
        assert _log_odds(y01, w) == pytest.approx(np.log(3.0))

    def test_prior_logits_softmax_back_to_frequencies(self) -> None:
        y_onehot = np.eye(3)[np.array([0, 0, 1, 1, 1, 2])]
        w = np.ones(6)
        logits = _prior_logits(y_onehot, w)
        np.testing.assert_allclose(softmax(logits), [2 / 6, 3 / 6, 1 / 6], atol=1e-12)

    def test_binary_negative_gradient_is_y_minus_p(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.standard_normal(20)
        y01 = rng.integers(0, 2, size=20).astype(float)
        np.testing.assert_allclose(
            _binary_negative_gradient(y01, raw), y01 - sigmoid(raw)
        )

    def test_multinomial_negative_gradient_is_Y_minus_p(self) -> None:
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((15, 4))
        y_onehot = np.eye(4)[rng.integers(0, 4, size=15)]
        np.testing.assert_allclose(
            _multinomial_negative_gradient(y_onehot, raw),
            y_onehot - softmax(raw, axis=1),
        )

    def test_pseudo_hessian_equals_p_times_1_minus_p_on_onehot(self) -> None:
        p = np.array([0.1, 0.5, 0.9, 0.3])
        for y in (0.0, 1.0):
            r = y - p
            np.testing.assert_allclose(_pseudo_hessian(r), p * (1.0 - p))

    def test_newton_leaf_value_hand(self) -> None:
        g = np.array([0.4, -0.2, 0.1])
        h = np.array([0.24, 0.16, 0.09])
        w = np.ones(3)
        assert _newton_leaf_value(g, h, w, 1.0) == pytest.approx(g.sum() / h.sum())
        assert _newton_leaf_value(g, h, w, 2 / 3) == pytest.approx(
            (2 / 3) * g.sum() / h.sum()
        )

    def test_newton_leaf_value_zero_when_hessian_underflows(self) -> None:
        g = np.array([1e-9, -1e-9])
        h = np.array([1e-18, 1e-18])
        assert _newton_leaf_value(g, h, np.ones(2), 1.0) == 0.0


class TestOneRoundIndependently:
    def test_leaf_values_and_scores_match_a_hand_newton_step(self) -> None:
        X, y = make_moons(n_samples=160, noise=0.2, random_state=1)
        model = GradientBoostingClassifier(
            n_estimators=1, learning_rate=1.0, max_depth=2, init="zero"
        ).fit(X, y)
        tree = model.estimators_[0][0]

        y01 = (y == model.classes_[1]).astype(float)
        p0 = sigmoid(np.zeros(len(y)))  # init="zero"
        r = y01 - p0
        h = np.abs(r) * (1.0 - np.abs(r))

        for leaf, idx in _leaf_groups(tree, X):
            gamma = r[idx].sum() / h[idx].sum()
            assert leaf.value == pytest.approx(gamma)

        np.testing.assert_allclose(model.decision_function(X), tree.predict(X))

    def test_round_two_is_fit_to_the_recomputed_residuals(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.25, random_state=2)
        one = GradientBoostingClassifier(
            n_estimators=1, learning_rate=0.5, max_depth=2, init="zero"
        ).fit(X, y)
        raw_after_one = one.decision_function(X)

        two = GradientBoostingClassifier(
            n_estimators=2, learning_rate=0.5, max_depth=2, init="zero"
        ).fit(X, y)
        y01 = (y == two.classes_[1]).astype(float)
        expected_residual = y01 - sigmoid(raw_after_one)

        # round 2's tree is fit to exactly those residuals under squared_error,
        # so an independent DecisionTreeRegressor on them finds the same split
        refit = DecisionTreeRegressor(max_depth=2).fit(X, expected_residual)
        second_tree = two.estimators_[1][0]
        assert second_tree.tree_.feature == refit.tree_.feature
        np.testing.assert_allclose(second_tree.tree_.threshold, refit.tree_.threshold)


class TestSampleWeight:
    def test_integer_weights_match_row_duplication(self) -> None:
        X, y = make_blobs(
            n_samples=90, n_features=3, centers=2, cluster_std=4.0, random_state=1
        )
        rng = np.random.default_rng(3)
        w = rng.integers(1, 4, size=len(y)).astype(float)
        repeat = np.repeat(np.arange(len(y)), w.astype(int))

        weighted = GradientBoostingClassifier(
            n_estimators=15, max_depth=2, random_state=0
        ).fit(X, y, sample_weight=w)
        duplicated = GradientBoostingClassifier(
            n_estimators=15, max_depth=2, random_state=0
        ).fit(X[repeat], y[repeat])

        np.testing.assert_allclose(
            weighted.decision_function(X), duplicated.decision_function(X), atol=1e-9
        )

    def test_zero_weight_row_does_not_change_the_fit(self) -> None:
        X, y = make_blobs(
            n_samples=80, n_features=3, centers=2, cluster_std=4.0, random_state=2
        )
        base = GradientBoostingClassifier(
            n_estimators=10, max_depth=2, random_state=0
        ).fit(X, y)

        X_aug = np.vstack([X, X[0] + 5.0])
        y_aug = np.append(y, 1 - y[0])
        w_aug = np.append(np.ones(len(y)), 0.0)
        with_ghost = GradientBoostingClassifier(
            n_estimators=10, max_depth=2, random_state=0
        ).fit(X_aug, y_aug, sample_weight=w_aug)

        np.testing.assert_allclose(
            base.decision_function(X), with_ghost.decision_function(X), atol=1e-9
        )


class TestBehaviour:
    def test_training_loss_is_non_increasing_binary(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
        model = GradientBoostingClassifier(
            n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X, y)
        assert np.all(np.diff(model.train_score_) <= 1e-9)
        assert model.train_score_.shape == (60,)

    def test_training_loss_is_non_increasing_multiclass(self) -> None:
        X, y = make_blobs(
            n_samples=360, n_features=4, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(
            n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X, y)
        assert np.all(np.diff(model.train_score_) <= 1e-9)

    def test_beats_a_shallow_tree_on_held_out_data(self) -> None:
        X, y = make_moons(n_samples=800, noise=0.3, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        # a single shallow tree underfits interleaved moons
        stump_tree = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
        depth3 = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr)
        boosted = GradientBoostingClassifier(
            n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        assert boosted.score(X_te, y_te) > stump_tree.score(X_te, y_te) + 0.05
        assert boosted.score(X_te, y_te) >= depth3.score(X_te, y_te) - 0.01

    def test_staged_predict_trends_upward(self) -> None:
        X, y = make_moons(n_samples=600, noise=0.3, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        model = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        accs = [np.mean(p == y_te) for p in model.staged_predict(X_te)]
        assert len(accs) == 120
        assert np.mean(accs[-20:]) > np.mean(accs[:20])

    def test_smaller_learning_rate_overfits_training_slower(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.3, random_state=2)
        fast = GradientBoostingClassifier(
            n_estimators=40, learning_rate=1.0, max_depth=3, random_state=0
        ).fit(X, y)
        slow = GradientBoostingClassifier(
            n_estimators=40, learning_rate=0.05, max_depth=3, random_state=0
        ).fit(X, y)
        assert fast.score(X, y) >= slow.score(X, y)

    def test_shrinkage_traded_for_more_rounds_reaches_similar_accuracy(self) -> None:
        X, y = make_moons(n_samples=800, noise=0.3, random_state=3)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        a = GradientBoostingClassifier(
            n_estimators=60, learning_rate=0.5, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        b = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        assert abs(a.score(X_te, y_te) - b.score(X_te, y_te)) < 0.06


class TestSubsample:
    def test_reproducible_under_a_fixed_seed(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
        a = GradientBoostingClassifier(
            n_estimators=30, subsample=0.6, random_state=42
        ).fit(X, y)
        b = GradientBoostingClassifier(
            n_estimators=30, subsample=0.6, random_state=42
        ).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))

    def test_differs_from_full_sample_and_still_fits(self) -> None:
        X, y = make_moons(n_samples=500, noise=0.3, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        full = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        stoch = GradientBoostingClassifier(
            n_estimators=80, subsample=0.5, max_depth=3, random_state=0
        ).fit(X_tr, y_tr)
        assert not np.allclose(
            full.decision_function(X_te), stoch.decision_function(X_te)
        )
        assert stoch.score(X_te, y_te) > 0.6


class TestInit:
    def test_zero_init_sets_the_score_to_zeros(self) -> None:
        X, y = make_blobs(
            n_samples=120, n_features=3, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(n_estimators=5, init="zero").fit(X, y)
        np.testing.assert_array_equal(model.init_score_, np.zeros(3))

    def test_prior_init_binary_is_the_log_odds(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(n_estimators=5).fit(X, y)
        y01 = (y == model.classes_[1]).astype(float)
        assert model.init_score_[0] == pytest.approx(_log_odds(y01, np.ones(len(y))))

    def test_prior_init_multiclass_softmaxes_to_class_frequencies(self) -> None:
        X, y = make_blobs(
            n_samples=180, n_features=3, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(n_estimators=5).fit(X, y)
        freqs = np.array([(y == c).mean() for c in model.classes_])
        np.testing.assert_allclose(softmax(model.init_score_), freqs, atol=1e-12)


class TestPredictionOutputs:
    def test_decision_function_shapes(self) -> None:
        Xb, yb = make_moons(n_samples=150, noise=0.2, random_state=0)
        binary = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(Xb, yb)
        assert binary.decision_function(Xb).shape == (150,)

        Xm, ym = make_blobs(
            n_samples=150, n_features=3, centers=3, cluster_std=3.0, random_state=0
        )
        multi = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(Xm, ym)
        assert multi.decision_function(Xm).shape == (150, 3)

    def test_predict_proba_is_a_distribution(self) -> None:
        X, y = make_blobs(
            n_samples=200, n_features=4, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_predict_is_argmax_of_proba(self) -> None:
        X, y = make_blobs(
            n_samples=180, n_features=3, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
        expected = model.classes_[np.argmax(model.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(model.predict(X), expected)

    def test_binary_decision_function_sign_agrees_with_predict(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
        model = GradientBoostingClassifier(n_estimators=40, random_state=0).fit(X, y)
        positive = model.decision_function(X) > 0.0
        np.testing.assert_array_equal(
            model.classes_[positive.astype(int)], model.predict(X)
        )

    def test_staged_iterators_end_at_the_full_ensemble(self) -> None:
        X, y = make_blobs(
            n_samples=150, n_features=3, centers=3, cluster_std=3.0, random_state=0
        )
        model = GradientBoostingClassifier(n_estimators=15, random_state=0).fit(X, y)
        preds = list(model.staged_predict(X))
        probas = list(model.staged_predict_proba(X))
        decisions = list(model.staged_decision_function(X))
        assert len(preds) == 15 and len(probas) == 15 and len(decisions) == 15
        np.testing.assert_array_equal(preds[-1], model.predict(X))
        np.testing.assert_allclose(probas[-1], model.predict_proba(X))
        np.testing.assert_allclose(decisions[-1], model.decision_function(X))

    def test_staged_decision_function_binary_shape(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(n_estimators=8, random_state=0).fit(X, y)
        stages = list(model.staged_decision_function(X))
        assert len(stages) == 8
        assert all(stage.shape == (200,) for stage in stages)
        np.testing.assert_allclose(stages[-1], model.decision_function(X))

    def test_feature_importances(self) -> None:
        rng = np.random.default_rng(0)
        X_signal, y = make_blobs(
            n_samples=300, n_features=2, centers=2, cluster_std=3.0, random_state=0
        )
        X = np.column_stack([X_signal, rng.standard_normal(300)])  # 3rd = noise
        model = GradientBoostingClassifier(
            n_estimators=40, max_depth=3, random_state=0
        ).fit(X, y)
        assert model.feature_importances_.shape == (3,)
        assert model.feature_importances_.sum() == pytest.approx(1.0)
        assert model.feature_importances_[2] < 0.15


class TestDeterminism:
    def test_same_seed_same_decision_function(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
        a = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
        b = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))

    def test_deterministic_path_needs_no_seed(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
        a = GradientBoostingClassifier(n_estimators=20).fit(X, y)
        b = GradientBoostingClassifier(n_estimators=20).fit(X, y)
        np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GradientBoostingClassifier().predict(np.zeros((2, 2)))

    def test_predict_proba_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GradientBoostingClassifier().predict_proba(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_moons(n_samples=80, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(n_estimators=5)
        assert model.fit(X, y) is model

    def test_hyperparameters_not_mutated(self) -> None:
        X, y = make_moons(n_samples=80, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(
            n_estimators=5, learning_rate=0.2, subsample=0.8, random_state=0
        )
        before = model.get_params()
        model.fit(X, y)
        assert model.get_params() == before

    def test_feature_count_mismatch_raises(self) -> None:
        X, y = make_moons(n_samples=80, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(n_estimators=5).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((3, 5)))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_estimators": 0}, "n_estimators"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"learning_rate": -0.1}, "learning_rate"),
            ({"subsample": 0.0}, "subsample"),
            ({"subsample": 1.5}, "subsample"),
            ({"max_depth": 0}, "max_depth"),
            ({"init": "mean"}, "init"),
        ],
    )
    def test_bad_hyperparameters_raise(self, kwargs, match) -> None:
        X, y = make_moons(n_samples=60, noise=0.2, random_state=0)
        with pytest.raises(ValueError, match=match):
            GradientBoostingClassifier(**kwargs).fit(X, y)

    def test_single_class_target_raises(self) -> None:
        X = np.random.default_rng(0).standard_normal((20, 3))
        with pytest.raises(ValueError, match="classes"):
            GradientBoostingClassifier(n_estimators=5).fit(X, np.zeros(20))

    def test_bad_sample_weight_raises(self) -> None:
        X, y = make_moons(n_samples=40, noise=0.2, random_state=0)
        with pytest.raises(ValueError):
            GradientBoostingClassifier(n_estimators=5).fit(
                X, y, sample_weight=-np.ones(40)
            )

    def test_repr_round_trips(self) -> None:
        model = GradientBoostingClassifier(n_estimators=7, learning_rate=0.3)
        assert "n_estimators=7" in repr(model)
        assert "learning_rate=0.3" in repr(model)


class TestEdges:
    def test_single_feature(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 1))
        y = (X[:, 0] > 0).astype(int)
        model = GradientBoostingClassifier(
            n_estimators=20, max_depth=1, random_state=0
        ).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_n_estimators_one(self) -> None:
        X, y = make_moons(n_samples=100, noise=0.2, random_state=0)
        model = GradientBoostingClassifier(n_estimators=1, random_state=0).fit(X, y)
        assert len(model.estimators_) == 1
        assert model.train_score_.shape == (1,)

    def test_max_features_uses_the_rng(self) -> None:
        X, y = make_blobs(
            n_samples=200, n_features=6, centers=2, cluster_std=4.0, random_state=0
        )
        a = GradientBoostingClassifier(
            n_estimators=20, max_features="sqrt", random_state=0
        ).fit(X, y)
        b = GradientBoostingClassifier(
            n_estimators=20, max_features="sqrt", random_state=1
        ).fit(X, y)
        assert not np.allclose(a.decision_function(X), b.decision_function(X))
