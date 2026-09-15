"""Tests for scratchgrad.svm.LinearSVM.

Tiers (plan.md section 3): an analytic check at (w, b) = 0, a
finite-difference subgradient check away from any margin kink, a convexity
(Jensen) check on the pure objective, a best-iterate-tracking check (the
non-smooth-objective analogue of a monotone-decrease check), determinism,
the fit/predict contract, behavioral checks on synthetic data, and edge
cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.exceptions import NotFittedError
from scratchgrad.metrics import accuracy_score
from scratchgrad.preprocessing import train_test_split
from scratchgrad.svm import LinearSVM
from scratchgrad.svm.linear_svm import _svm_objective, _svm_subgradient
from tests.helpers.gradcheck import gradient_check


class TestAnalytic:
    def test_objective_and_subgradient_at_zero(self) -> None:
        # At w=b=0 every margin is 0 < 1, so all samples are violated:
        # J = C*n, g_w = -C * X^T y, g_b = -C * sum(y).
        X = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
        y = np.array([1.0, -1.0, 1.0])
        C = 2.0
        w, b = np.zeros(2), 0.0

        assert _svm_objective(X, y, w, b, C) == pytest.approx(C * 3)
        g_w, g_b = _svm_subgradient(X, y, w, b, C)
        assert g_w == pytest.approx(-C * (X.T @ y))
        assert g_b == pytest.approx(-C * y.sum())


class TestGradient:
    @pytest.mark.parametrize("C", [0.3, 1.0, 5.0])
    def test_subgradient_matches_finite_differences(self, rng, C) -> None:
        X = rng.standard_normal((20, 4))
        y = np.where(rng.random(20) < 0.5, -1.0, 1.0)
        w = 0.3 * rng.standard_normal(4)
        b = 0.2
        theta = np.concatenate([w, [b]])

        g_w, g_b = _svm_subgradient(X, y, w, b, C)
        analytic = np.concatenate([g_w, [g_b]])
        gradient_check(
            lambda t: _svm_objective(X, y, t[:-1], t[-1], C), analytic, theta
        )


class TestConvexity:
    def test_objective_is_convex(self, rng) -> None:
        X = rng.standard_normal((15, 3))
        y = np.where(rng.random(15) < 0.5, -1.0, 1.0)
        w1, b1 = rng.standard_normal(3), 0.4
        w2, b2 = rng.standard_normal(3), -0.3
        C = 1.0

        for lam in (0.25, 0.5, 0.75):
            w_mix = lam * w1 + (1 - lam) * w2
            b_mix = lam * b1 + (1 - lam) * b2
            lhs = _svm_objective(X, y, w_mix, b_mix, C)
            rhs = lam * _svm_objective(X, y, w1, b1, C) + (1 - lam) * _svm_objective(
                X, y, w2, b2, C
            )
            assert lhs <= rhs + 1e-9


class TestBestIterate:
    def test_more_steps_never_worsens_the_kept_objective(self) -> None:
        # The subgradient method's best-J tracking is a superset relation:
        # every step max_iter=50 sees is also seen by max_iter=500, so the
        # kept objective can only stay the same or improve.
        X, y = make_moons(n_samples=150, noise=0.3, random_state=0)

        short = LinearSVM(lr=0.5, max_iter=50).fit(X, y)
        long = LinearSVM(lr=0.5, max_iter=500).fit(X, y)

        target = np.where(y == short.classes_[1], 1.0, -1.0)
        J_short = _svm_objective(X, target, short.coef_, short.intercept_, short.C)
        J_long = _svm_objective(X, target, long.coef_, long.intercept_, long.C)
        assert J_long <= J_short + 1e-9

    def test_kept_objective_beats_the_zero_start(self) -> None:
        X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=1)
        model = LinearSVM(lr=1.0, max_iter=200).fit(X, y)
        target = np.where(y == model.classes_[1], 1.0, -1.0)

        J_fitted = _svm_objective(X, target, model.coef_, model.intercept_, model.C)
        J_start = _svm_objective(X, target, np.zeros(X.shape[1]), 0.0, model.C)
        assert J_fitted <= J_start


class TestDeterminism:
    def test_two_fits_are_byte_identical(self) -> None:
        X, y = make_moons(n_samples=100, noise=0.25, random_state=2)
        a = LinearSVM().fit(X, y)
        b = LinearSVM().fit(X, y)
        np.testing.assert_array_equal(a.coef_, b.coef_)
        assert a.intercept_ == b.intercept_


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            LinearSVM().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = LinearSVM()
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = LinearSVM(C=0.5, fit_intercept=False, lr=0.02, max_iter=50)
        model.fit(X, y)
        assert model.get_params() == {
            "C": 0.5,
            "fit_intercept": False,
            "lr": 0.02,
            "max_iter": 50,
        }

    def test_coef_shape_matches_n_features(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=5, centers=2, random_state=0)
        model = LinearSVM().fit(X, y)
        assert model.coef_.shape == (5,)

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = LinearSVM().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_non_binary_target_raises(self) -> None:
        X, y = make_blobs(n_samples=60, centers=3, random_state=0)
        with pytest.raises(ValueError, match="binary"):
            LinearSVM().fit(X, y)

    def test_non_positive_C_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="C must be > 0"):
            LinearSVM(C=0.0).fit(X, y)

    def test_non_positive_lr_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="lr must be > 0"):
            LinearSVM(lr=0.0).fit(X, y)

    def test_max_iter_below_one_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="max_iter"):
            LinearSVM(max_iter=0).fit(X, y)

    def test_n_iter_equals_max_iter(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        model = LinearSVM(max_iter=37).fit(X, y)
        assert model.n_iter_ == 37

    def test_no_predict_proba(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        model = LinearSVM().fit(X, y)
        assert not hasattr(model, "predict_proba")

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        model = LinearSVM().fit(X, y)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, model.predict(X)))

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(LinearSVM()) == (
            "LinearSVM(C=1.0, fit_intercept=True, lr=0.01, max_iter=1000)"
        )


class TestBehavioral:
    def test_separable_blobs_are_classified_almost_perfectly(self) -> None:
        X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
        model = LinearSVM().fit(X, y)
        assert model.score(X, y) > 0.95

    def test_beats_majority_baseline_on_noisy_moons(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.2, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        model = LinearSVM(max_iter=2000).fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.8

    def test_larger_C_does_not_shrink_the_weight_norm(self) -> None:
        # Larger C weighs margin violations more heavily relative to
        # ||w||, which pushes toward a tighter (larger-norm) fit on
        # separable data -- the opposite of shrinking toward 0.
        X, y = make_blobs(n_samples=150, centers=2, cluster_std=1.0, random_state=3)
        soft = LinearSVM(C=0.01, max_iter=2000).fit(X, y)
        hard = LinearSVM(C=10.0, max_iter=2000).fit(X, y)
        assert np.linalg.norm(hard.coef_) >= np.linalg.norm(soft.coef_)


class TestEdgeCases:
    def test_fit_intercept_false(self) -> None:
        X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.0, random_state=4)
        X = X - X.mean(axis=0)
        model = LinearSVM(fit_intercept=False).fit(X, y)
        assert model.intercept_ == 0.0
        assert model.coef_.shape == (X.shape[1],)

    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=1, centers=2, random_state=5)
        model = LinearSVM().fit(X, y)
        assert model.coef_.shape == (1,)

    def test_perfectly_separable_toy_dataset_recovers_correct_sign(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        model = LinearSVM().fit(X, y)
        assert model.coef_[0] > 0
        assert model.predict(X).tolist() == [0.0, 0.0, 1.0, 1.0]

    def test_predict_labels_are_drawn_from_classes(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, cluster_std=1.0, random_state=8)
        model = LinearSVM().fit(X, 2 * y - 1)  # labels {-1, 1}
        assert model.classes_.tolist() == [-1.0, 1.0]
        assert set(np.unique(model.predict(X))).issubset({-1.0, 1.0})
