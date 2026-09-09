"""Tests for scratchgrad.ensemble.AdaBoostClassifier (SAMME).

Tiers (plan.md section 3): AdaBoost minimises a multi-class exponential
loss but there is no gradient to check, so the correctness analogues are
(a) the one-round weight update recomputed independently, (b) a K=2 run
reproducing a from-scratch classic discrete AdaBoost, and (c) reducing to
the lone base stump at n_estimators=1. Plus the analytic alpha/error
helpers, boosting behaviour, the early-stop rules, decision_function /
predict_proba, staged prediction, the fit/predict contract, and edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import AdaBoostClassifier
from scratchgrad.ensemble.adaboost import (
    _samme_alpha,
    _samme_decision,
    _weighted_error,
)
from scratchgrad.exceptions import NotFittedError
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier


def _classic_discrete_adaboost(X, y, n_rounds):
    """A deliberately naive binary AdaBoost (Freund-Schapire), for K=2 parity.

    Uses the same weighted stump as the estimator under test; returns the
    Freund-Schapire estimator weights (half the SAMME ones) and the vote
    predictions on the training set.
    """
    classes = np.unique(y)
    y_pm = np.where(y == classes[1], 1.0, -1.0)  # +-1 labels
    w = np.full(len(y), 1.0 / len(y))
    alphas, stump_pm = [], []
    for _ in range(n_rounds):
        stump = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        pred = stump.predict(X)
        pred_pm = np.where(pred == classes[1], 1.0, -1.0)
        err = float(w[pred != y].sum())
        alpha = 0.5 * np.log((1.0 - err) / err)
        w = w * np.exp(-alpha * y_pm * pred_pm)
        w = w / w.sum()
        alphas.append(alpha)
        stump_pm.append(pred_pm)
    margin = sum(a * p for a, p in zip(alphas, stump_pm, strict=True))
    vote = np.where(margin >= 0.0, classes[1], classes[0])
    return np.array(alphas), vote


class TestHelpers:
    def test_samme_alpha_matches_formula(self) -> None:
        assert _samme_alpha(0.25, 3, 1.0) == pytest.approx(
            np.log(0.75 / 0.25) + np.log(2)
        )

    def test_samme_alpha_binary_drops_the_log_term(self) -> None:
        assert _samme_alpha(0.2, 2, 1.0) == pytest.approx(np.log(0.8 / 0.2))

    def test_learning_rate_scales_alpha_linearly(self) -> None:
        base = _samme_alpha(0.3, 4, 1.0)
        assert _samme_alpha(0.3, 4, 0.5) == pytest.approx(0.5 * base)

    def test_weighted_error_on_a_hand_example(self) -> None:
        y = np.array([0.0, 1.0, 0.0, 1.0])
        pred = np.array([0.0, 0.0, 0.0, 1.0])  # row 1 wrong
        w = np.array([0.1, 0.4, 0.3, 0.2])
        assert _weighted_error(y, pred, w) == pytest.approx(0.4)

    def test_samme_decision_rows_sum_to_zero(self) -> None:
        X, y = make_blobs(n_samples=120, centers=3, random_state=0)
        model = AdaBoostClassifier(n_estimators=8).fit(X, y)
        score = _samme_decision(
            model.estimators_, model.estimator_weights_, model.classes_, X[:10]
        )
        np.testing.assert_allclose(score.sum(axis=1), 0.0, atol=1e-12)


class TestWeightUpdate:
    def test_one_round_update_recomputed_independently(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=1).fit(X, y)
        stump = model.estimators_[0]
        err = model.estimator_errors_[0]
        alpha = model.estimator_weights_[0]

        w0 = np.full(len(y), 1.0 / len(y))
        miss = stump.predict(X) != y
        expected = w0 * np.exp(alpha * miss)
        expected = expected / expected.sum()

        # refit round 2 and read the weights it trained against, indirectly:
        # the second stump must be the one a weighted fit on `expected` gives.
        model2 = AdaBoostClassifier(n_estimators=2).fit(X, y)
        ref = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=expected)
        np.testing.assert_array_equal(model2.estimators_[1].predict(X), ref.predict(X))
        assert err == pytest.approx(_weighted_error(y, stump.predict(X), w0))


class TestReducesToClassicAdaBoost:
    def test_binary_run_matches_from_scratch_discrete_adaboost(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.3, random_state=1)
        rounds = 15
        model = AdaBoostClassifier(n_estimators=rounds).fit(X, y)
        assert len(model.estimators_) == rounds  # no early stop on this data

        fs_alphas, fs_vote = _classic_discrete_adaboost(X, y, rounds)
        np.testing.assert_allclose(model.estimator_weights_, 2.0 * fs_alphas, rtol=1e-9)
        np.testing.assert_array_equal(model.predict(X), fs_vote)


class TestReducesToBase:
    def test_one_estimator_equals_a_lone_stump(self) -> None:
        X, y = make_blobs(n_samples=200, centers=3, cluster_std=3.0, random_state=2)
        model = AdaBoostClassifier(n_estimators=1).fit(X, y)
        # not early-stopped-on-perfection: this data isn't stump-separable
        assert model.estimator_errors_[0] > 0.0
        stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), stump.predict(X))


class TestBoostingBehaviour:
    def test_ensemble_beats_a_single_stump_on_moons(self) -> None:
        X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
        boosted = AdaBoostClassifier(n_estimators=100).fit(X_tr, y_tr)
        assert boosted.score(X_te, y_te) > stump.score(X_te, y_te) + 0.1

    def test_staged_train_accuracy_trends_up(self) -> None:
        X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=60).fit(X, y)
        staged = list(model.staged_score(X, y))
        assert staged[-1] >= staged[0]
        # later half is on average at least as good as the first few rounds
        assert np.mean(staged[30:]) >= np.mean(staged[:5])

    def test_multiclass_high_accuracy(self) -> None:
        X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
        model = AdaBoostClassifier(n_estimators=40).fit(X_tr, y_tr)
        assert model.score(X_te, y_te) > 0.9


class TestEarlyStop:
    def test_perfectly_separable_stops_after_one_round(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = AdaBoostClassifier(n_estimators=50).fit(X, y)
        assert len(model.estimators_) == 1
        assert model.estimator_errors_[0] == 0.0
        assert model.estimator_weights_[0] == 1.0

    def test_first_round_worse_than_random_raises(self) -> None:
        # XOR: no stump beats 50% -> err >= 1 - 1/2 on round 1.
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])
        with pytest.raises(ValueError, match="random guessing"):
            AdaBoostClassifier(n_estimators=10).fit(X, y)


class TestDecisionAndProba:
    def test_binary_decision_function_is_1d_and_agrees_with_predict(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=30).fit(X, y)
        df = model.decision_function(X)
        assert df.shape == (200,)
        agree = (df >= 0) == (model.predict(X) == model.classes_[1])
        assert agree.all()

    def test_multiclass_decision_function_is_2d(self) -> None:
        X, y = make_blobs(n_samples=150, centers=3, random_state=0)
        model = AdaBoostClassifier(n_estimators=20).fit(X, y)
        assert model.decision_function(X).shape == (150, 3)

    def test_proba_is_a_distribution_and_argmax_is_predict(self) -> None:
        X, y = make_blobs(n_samples=150, centers=3, cluster_std=2.5, random_state=0)
        model = AdaBoostClassifier(n_estimators=25).fit(X, y)
        proba = model.predict_proba(X[:20])
        assert proba.shape == (20, 3)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(20))
        labels = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X[:20]), labels)


class TestStaged:
    def test_staged_predict_length_and_final(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=25).fit(X, y)
        staged = list(model.staged_predict(X))
        assert len(staged) == len(model.estimators_)
        np.testing.assert_array_equal(staged[-1], model.predict(X))


class TestFeatureImportances:
    def test_sum_to_one_and_zero_for_an_unused_feature(self, rng) -> None:
        x0 = rng.standard_normal(300)
        y = (x0 > 0.0).astype(np.float64)
        X = np.column_stack([x0, rng.standard_normal(300)])  # feature 1 is noise
        model = AdaBoostClassifier(n_estimators=20).fit(X, y)
        assert model.feature_importances_.sum() == pytest.approx(1.0)
        assert model.feature_importances_[0] > 0.9

    def test_is_the_alpha_weighted_mean_of_stump_importances(self) -> None:
        X, y = make_blobs(n_samples=200, n_features=4, centers=3, random_state=0)
        model = AdaBoostClassifier(n_estimators=15).fit(X, y)
        w = model.estimator_weights_ / model.estimator_weights_.sum()
        stacked = np.array([t.feature_importances_ for t in model.estimators_])
        expected = w @ stacked
        expected = expected / expected.sum()
        np.testing.assert_allclose(model.feature_importances_, expected)


class TestDeterminism:
    def test_same_data_same_fit(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
        a = AdaBoostClassifier(n_estimators=20).fit(X, y)
        b = AdaBoostClassifier(n_estimators=20).fit(X, y)
        np.testing.assert_array_equal(a.estimator_weights_, b.estimator_weights_)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            AdaBoostClassifier().predict(np.zeros((2, 2)))

    def test_decision_function_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            AdaBoostClassifier().decision_function(np.zeros((2, 2)))

    def test_staged_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            list(AdaBoostClassifier().staged_predict(np.zeros((2, 2))))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, random_state=0)
        model = AdaBoostClassifier(n_estimators=5)
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=3.0, random_state=0)
        model = AdaBoostClassifier(n_estimators=10, learning_rate=0.5, max_depth=2)
        model.fit(X, y)
        assert model.get_params() == {
            "n_estimators": 10,
            "learning_rate": 0.5,
            "max_depth": 2,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=60, n_features=3, centers=2, random_state=0)
        model = AdaBoostClassifier(n_estimators=5).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="features"):
            list(model.staged_predict(np.zeros((4, 2))))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_estimators": 0}, "n_estimators must be >= 1"),
            ({"learning_rate": 0.0}, "learning_rate must be > 0"),
            ({"learning_rate": -1.0}, "learning_rate must be > 0"),
            ({"max_depth": 0}, "max_depth must be >= 1"),
        ],
    )
    def test_bad_hyperparameters_raise(self, kwargs, match) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match=match):
            AdaBoostClassifier(**kwargs).fit(X, y)

    def test_bad_sample_weight_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="sample_weight"):
            AdaBoostClassifier().fit(X, y, sample_weight=np.ones(5))

    def test_repr_round_trips(self) -> None:
        assert repr(AdaBoostClassifier()) == (
            "AdaBoostClassifier(n_estimators=50, learning_rate=1.0, max_depth=1)"
        )

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=80, centers=2, cluster_std=3.0, random_state=0)
        model = AdaBoostClassifier(n_estimators=10).fit(X, y)
        assert model.score(X, y) == pytest.approx(np.mean(model.predict(X) == y))


class TestEdgeCases:
    def test_sample_weight_focuses_the_first_stump(self) -> None:
        X, y = make_blobs(n_samples=200, centers=2, cluster_std=4.0, random_state=0)
        w = np.where(y == 1, 5.0, 1.0)
        model = AdaBoostClassifier(n_estimators=1).fit(X, y, sample_weight=w)
        ref = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w / w.sum())
        np.testing.assert_array_equal(model.predict(X), ref.predict(X))

    def test_learning_rate_below_one_still_fits(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=50, learning_rate=0.5).fit(X, y)
        assert model.score(X, y) > 0.8

    def test_deeper_base_learner(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.35, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        model = AdaBoostClassifier(n_estimators=20, max_depth=3).fit(X_tr, y_tr)
        assert all(e.max_depth_ <= 3 for e in model.estimators_)
        assert model.score(X_te, y_te) > 0.8
