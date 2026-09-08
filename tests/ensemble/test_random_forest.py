"""Tests for scratchgrad.ensemble.RandomForestClassifier.

Tiers (plan.md section 3): a random forest has no objective and no
gradient, so the correctness analogues are (a) it reduces exactly to a
single tree when the randomness is switched off, (b) `predict_proba` is an
independent re-average of the per-tree probabilities, and (c) the
out-of-bag score tracks a real held-out score. Plus the variance-reduction
behaviour that motivates the method, bootstrap statistics, determinism,
the fit/predict contract, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import RandomForestClassifier
from scratchgrad.ensemble.random_forest import _aggregate_proba, _oob_score
from scratchgrad.exceptions import NotFittedError
from scratchgrad.linear import LogisticRegression
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier


def _tree_equal(a, b):
    if a.is_leaf or b.is_leaf:
        return a.is_leaf and b.is_leaf and np.allclose(a.value, b.value)
    return (
        a.feature == b.feature
        and a.threshold == b.threshold
        and _tree_equal(a.left, b.left)
        and _tree_equal(a.right, b.right)
    )


class TestReducesToASingleTree:
    def test_one_tree_no_bootstrap_no_subsampling_equals_lone_tree(self) -> None:
        X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=1)
        forest = RandomForestClassifier(
            n_estimators=1, bootstrap=False, max_features=None, random_state=5
        ).fit(X, y)
        tree = DecisionTreeClassifier(random_state=5).fit(X, y)
        assert _tree_equal(forest.estimators_[0].tree_, tree.tree_)
        np.testing.assert_array_equal(forest.predict(X), tree.predict(X))

    def test_no_bootstrap_no_subsampling_makes_every_tree_identical(self) -> None:
        X, y = make_blobs(n_samples=120, centers=3, random_state=0)
        forest = RandomForestClassifier(
            n_estimators=5, bootstrap=False, max_features=None, random_state=0
        ).fit(X, y)
        first = forest.estimators_[0].tree_
        assert all(_tree_equal(t.tree_, first) for t in forest.estimators_[1:])


class TestAggregation:
    def test_predict_proba_is_the_mean_of_the_trees(self) -> None:
        X, y = make_blobs(n_samples=200, n_features=3, centers=3, random_state=0)
        forest = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, y)
        manual = np.mean([t.predict_proba(X[:20]) for t in forest.estimators_], axis=0)
        np.testing.assert_allclose(forest.predict_proba(X[:20]), manual)

    def test_rows_sum_to_one_and_agree_with_predict(self) -> None:
        X, y = make_blobs(n_samples=150, centers=4, cluster_std=2.0, random_state=0)
        forest = RandomForestClassifier(n_estimators=10, random_state=1).fit(X, y)
        proba = forest.predict_proba(X[:25])
        assert proba.shape == (25, 4)
        assert proba.sum(axis=1) == pytest.approx(np.ones(25))
        labels = forest.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(forest.predict(X[:25]), labels)

    def test_aggregate_handles_a_tree_missing_a_class(self) -> None:
        # forest sees 3 classes; one tree is fit on a slice with only {0, 1}
        X, y = make_blobs(n_samples=180, n_features=2, centers=3, random_state=0)
        classes = np.unique(y)
        two_class = y < 2
        narrow = DecisionTreeClassifier(max_depth=3).fit(X[two_class], y[two_class])
        full = DecisionTreeClassifier(max_depth=3).fit(X, y)
        assert narrow.classes_.shape[0] == 2  # noqa: PLR2004

        proba = _aggregate_proba([narrow, full], classes, X[:10])
        assert proba.shape == (10, 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10))
        # the narrow tree contributes 0 to class 2, so that column is full/2
        np.testing.assert_allclose(proba[:, 2], full.predict_proba(X[:10])[:, 2] / 2)


class TestOutOfBag:
    def test_bootstrap_omission_probability(self) -> None:
        # each row is left out of a bootstrap of size n with prob ~ 1/e
        for n in (50, 500, 5000):
            assert (1 - 1 / n) ** n == pytest.approx(np.exp(-1), rel=0.05)

    def test_oob_score_tracks_a_real_holdout(self) -> None:
        X, y = make_blobs(n_samples=500, centers=3, cluster_std=3.0, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        forest = RandomForestClassifier(
            n_estimators=60, oob_score=True, random_state=0
        ).fit(X_tr, y_tr)
        assert abs(forest.oob_score_ - forest.score(X_te, y_te)) < 0.1

    def test_oob_decision_function_shape_and_rows(self) -> None:
        X, y = make_blobs(n_samples=200, centers=3, random_state=0)
        forest = RandomForestClassifier(
            n_estimators=50, oob_score=True, random_state=0
        ).fit(X, y)
        df = forest.oob_decision_function_
        assert df.shape == (200, 3)
        scored = ~np.isnan(df).any(axis=1)
        assert scored.mean() > 0.99  # almost every row is OOB for some tree
        np.testing.assert_allclose(df[scored].sum(axis=1), np.ones(scored.sum()))

    def test_oob_without_bootstrap_raises(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        with pytest.raises(ValueError, match="oob_score=True requires bootstrap"):
            RandomForestClassifier(oob_score=True, bootstrap=False).fit(X, y)

    def test_oob_attributes_absent_when_not_requested(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        assert not hasattr(forest, "oob_score_")
        assert not hasattr(forest, "oob_decision_function_")

    def test_oob_score_helper_directly(self) -> None:
        X, y = make_blobs(n_samples=120, n_features=2, centers=2, random_state=0)
        classes = np.unique(y)
        t0 = DecisionTreeClassifier(max_depth=3).fit(X, y)
        t1 = DecisionTreeClassifier(max_depth=3).fit(X, y)
        m0 = np.zeros(120, dtype=bool)
        m0[:60] = True  # t0 is OOB on the first half
        m1 = ~m0  # t1 is OOB on the second half
        decision, acc = _oob_score([t0, t1], [m0, m1], X, y, classes)
        assert decision.shape == (120, 2)
        assert not np.isnan(decision).any()  # every row covered by exactly one tree
        assert 0.0 <= acc <= 1.0

    def test_oob_score_skips_a_tree_that_saw_every_row(self) -> None:
        X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
        classes = np.unique(y)
        t0 = DecisionTreeClassifier(max_depth=2).fit(X, y)
        t1 = DecisionTreeClassifier(max_depth=2).fit(X, y)
        none_oob = np.zeros(100, dtype=bool)  # t0 was in-bag everywhere -> skipped
        all_oob = np.ones(100, dtype=bool)
        decision, _ = _oob_score([t0, t1], [none_oob, all_oob], X, y, classes)
        np.testing.assert_allclose(decision, t1.predict_proba(X))


class TestVarianceReduction:
    def test_forest_beats_a_single_deep_tree_on_noisy_data(self) -> None:
        X, y = make_moons(n_samples=600, noise=0.35, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(
            X_tr, y_tr
        )
        tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
        assert forest.score(X_te, y_te) >= tree.score(X_te, y_te)

    def test_more_trees_are_more_stable_across_seeds(self) -> None:
        # apples-to-apples: a lone bootstrapped tree (n_estimators=1) vs a
        # 40-tree forest, both re-seeded. Averaging shrinks the seed-to-seed
        # disagreement — the variance term rho-free part going to 0.
        X, y = make_moons(n_samples=400, noise=0.35, random_state=0)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

        def spread(n_estimators: int) -> float:
            preds = np.array(
                [
                    RandomForestClassifier(n_estimators=n_estimators, random_state=s)
                    .fit(X_tr, y_tr)
                    .predict(X_te)
                    for s in range(6)
                ]
            )
            return np.mean(
                [
                    (preds[i] != preds[j]).mean()
                    for i in range(6)
                    for j in range(i + 1, 6)
                ]
            )

        assert spread(40) < spread(1)

    def test_accuracy_does_not_fall_as_trees_are_added(self) -> None:
        X, y = make_blobs(n_samples=400, centers=3, cluster_std=4.0, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        accs = [
            RandomForestClassifier(n_estimators=b, random_state=0)
            .fit(X_tr, y_tr)
            .score(X_te, y_te)
            for b in (1, 5, 20, 50)
        ]
        assert accs[-1] >= accs[0] - 0.02  # non-decreasing within noise


class TestDeterminism:
    def test_same_seed_same_forest(self) -> None:
        X, y = make_blobs(n_samples=150, n_features=5, centers=3, random_state=0)
        a = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        b = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)
        assert all(
            _tree_equal(ta.tree_, tb.tree_)
            for ta, tb in zip(a.estimators_, b.estimators_, strict=True)
        )
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_different_seed_differs(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.4, random_state=0)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.4, random_state=0)
        a = RandomForestClassifier(n_estimators=15, random_state=1).fit(X_tr, y_tr)
        b = RandomForestClassifier(n_estimators=15, random_state=2).fit(X_tr, y_tr)
        # the ensembles are not identical: their soft votes differ somewhere
        assert not np.allclose(a.predict_proba(X_te), b.predict_proba(X_te))


class TestMaxFeaturesPassthrough:
    @pytest.mark.parametrize(
        ("max_features", "expected"),
        [("sqrt", 3), ("log2", 3), (2, 2), (0.5, 5), (None, 10)],
    )
    def test_trees_get_the_resolved_max_features(self, max_features, expected) -> None:
        X, y = make_blobs(n_samples=120, n_features=10, centers=2, random_state=0)
        forest = RandomForestClassifier(
            n_estimators=4, max_features=max_features, random_state=0
        ).fit(X, y)
        assert all(t.max_features_ == expected for t in forest.estimators_)

    def test_default_is_sqrt(self) -> None:
        X, y = make_blobs(n_samples=80, n_features=16, centers=2, random_state=0)
        forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        assert all(t.max_features_ == 4 for t in forest.estimators_)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RandomForestClassifier().predict(np.zeros((2, 2)))

    def test_predict_proba_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            RandomForestClassifier().predict_proba(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = RandomForestClassifier(n_estimators=3)
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = RandomForestClassifier(
            n_estimators=7, criterion="entropy", max_depth=4, max_features="log2"
        )
        model.fit(X, y)
        assert model.get_params() == {
            "n_estimators": 7,
            "criterion": "entropy",
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
            "max_features": "log2",
            "bootstrap": True,
            "oob_score": False,
            "random_state": None,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_bad_n_estimators_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_estimators must be >= 1"):
            RandomForestClassifier(n_estimators=0).fit(X, y)

    def test_bad_per_tree_hyperparameter_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="criterion"):
            RandomForestClassifier(n_estimators=3, criterion="mse").fit(X, y)

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.0, random_state=0)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert model.score(X, y) == pytest.approx(np.mean(preds == y))

    def test_repr_round_trips(self) -> None:
        assert repr(RandomForestClassifier()) == (
            "RandomForestClassifier(n_estimators=100, criterion='gini', "
            "max_depth=None, min_samples_split=2, min_samples_leaf=1, "
            "min_impurity_decrease=0.0, max_features='sqrt', bootstrap=True, "
            "oob_score=False, random_state=None)"
        )


class TestBehavioral:
    def test_beats_linear_baseline_on_moons(self) -> None:
        X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
        linear = LogisticRegression().fit(X_tr, y_tr)
        assert forest.score(X_te, y_te) > linear.score(X_te, y_te)

    def test_high_accuracy_on_well_separated_multiclass(self) -> None:
        X, y = make_blobs(n_samples=400, centers=4, cluster_std=1.5, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
        forest = RandomForestClassifier(n_estimators=40, random_state=1).fit(X_tr, y_tr)
        assert forest.score(X_te, y_te) > 0.9


class TestEdgeCases:
    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=60, n_features=1, centers=2, random_state=5)
        forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert forest.predict(X).shape == (60,)

    def test_single_class_target(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        y = np.zeros(30)
        forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        assert np.all(forest.predict(X) == 0.0)
        assert forest.predict_proba(X).shape == (30, 1)

    def test_n_estimators_one(self) -> None:
        X, y = make_blobs(n_samples=80, centers=2, random_state=0)
        forest = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
        assert len(forest.estimators_) == 1
        assert forest.score(X, y) > 0.8

    def test_feature_importances_sum_to_one(self) -> None:
        X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=0)
        forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        assert forest.feature_importances_.sum() == pytest.approx(1.0)
        assert forest.feature_importances_.shape == (4,)
