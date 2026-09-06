"""Tests for scratchgrad.linear.LogisticRegression.

Tiers (plan.md section 3): an analytic gradient check at theta=0, a
finite-difference gradient check and a Hessian check on the pure functions,
a convexity check across Newton steps, solver-consistency, a regularisation
check, the fit/predict contract, behavioral checks on synthetic data, and
edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.exceptions import ConvergenceWarning, NotFittedError
from scratchgrad.linear import LogisticRegression
from scratchgrad.linear.logistic_regression import (
    _logistic_gradient,
    _logistic_hessian,
    _logistic_objective,
)
from scratchgrad.metrics import accuracy_score
from scratchgrad.preprocessing import StandardScaler, train_test_split
from tests.helpers.gradcheck import gradient_check, numerical_gradient


def _theta_of(model: LogisticRegression) -> np.ndarray:
    """Rebuild the folded parameter vector [b, w] from a fitted model."""
    return np.concatenate([[model.intercept_], model.coef_])


class TestAnalytic:
    def test_gradient_at_zero_is_half_minus_y(self) -> None:
        # At theta=0, p == 0.5 everywhere, so grad = (1/n) X_aug^T (0.5 - y),
        # with no penalty contribution (D @ 0 == 0).
        X_aug = np.array([[1.0, 2.0], [1.0, -1.0], [1.0, 0.5], [1.0, 3.0]])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        mask = np.array([0.0, 1.0])
        expected = X_aug.T @ (0.5 - y) / 4.0
        got = _logistic_gradient(X_aug, y, np.zeros(2), inv_C=1.0, penalty_mask=mask)
        assert got == pytest.approx(expected, abs=1e-12)


class TestGradient:
    @pytest.mark.parametrize("inv_C", [0.0, 1.0 / 0.7])
    def test_gradient_matches_finite_differences(self, rng, inv_C) -> None:
        X_aug = rng.standard_normal((15, 4))
        y = rng.integers(0, 2, size=15).astype(np.float64)
        theta = 0.3 * rng.standard_normal(4)
        mask = np.array([0.0, 1.0, 1.0, 1.0])
        analytic = _logistic_gradient(X_aug, y, theta, inv_C, mask)
        gradient_check(
            lambda t: _logistic_objective(X_aug, y, t, inv_C, mask), analytic, theta
        )

    def test_hessian_matches_finite_differences_of_gradient(self, rng) -> None:
        X_aug = rng.standard_normal((12, 3))
        y = rng.integers(0, 2, size=12).astype(np.float64)
        theta = 0.3 * rng.standard_normal(3)
        inv_C = 1.0 / 0.5
        mask = np.array([0.0, 1.0, 1.0])
        hessian = _logistic_hessian(X_aug, theta, inv_C, mask)
        for i in range(3):
            column = numerical_gradient(
                lambda t, i=i: _logistic_gradient(X_aug, y, t, inv_C, mask)[i], theta
            )
            assert hessian[i] == pytest.approx(column, abs=1e-5)


class TestConvexity:
    @pytest.mark.filterwarnings("ignore::scratchgrad.exceptions.ConvergenceWarning")
    def test_objective_decreases_across_newton_steps(self) -> None:
        X, y = make_blobs(n_samples=120, centers=2, cluster_std=3.0, random_state=1)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        mask = np.concatenate([[0.0], np.ones(X.shape[1])])
        classes = np.unique(y)
        target = (y == classes[1]).astype(np.float64)

        objectives = []
        for max_iter in range(1, 8):
            model = LogisticRegression(max_iter=max_iter).fit(X, y)
            objectives.append(
                _logistic_objective(X_aug, target, _theta_of(model), 1.0, mask)
            )
        assert all(
            nxt <= cur + 1e-12
            for cur, nxt in zip(objectives[:-1], objectives[1:], strict=True)
        )


class TestSolverConsistency:
    def test_gd_converges_to_newton(self) -> None:
        X, y = make_blobs(n_samples=200, centers=2, cluster_std=3.0, random_state=2)
        X = StandardScaler().fit_transform(X)  # GD needs scaled features

        newton = LogisticRegression(solver="newton").fit(X, y)
        gd = LogisticRegression(solver="gd", lr=0.5, max_iter=200_000, tol=1e-11).fit(
            X, y
        )

        assert gd.coef_ == pytest.approx(newton.coef_, abs=1e-4)
        assert gd.intercept_ == pytest.approx(newton.intercept_, abs=1e-4)


class TestRegularisation:
    def test_no_penalty_gives_larger_weights_than_l2(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)

        l2 = LogisticRegression(penalty="l2", C=1.0).fit(X, y)
        unpenalised = LogisticRegression(penalty=None).fit(X, y)

        assert np.linalg.norm(unpenalised.coef_) > np.linalg.norm(l2.coef_)
        assert l2.score(X, y) > 0.8
        assert unpenalised.score(X, y) > 0.8


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            LogisticRegression().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = LogisticRegression()
        assert model.fit(X, y) is model

    @pytest.mark.filterwarnings("ignore::scratchgrad.exceptions.ConvergenceWarning")
    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = LogisticRegression(
            C=0.5, penalty=None, solver="gd", lr=0.05, max_iter=50
        )
        model.fit(X, y)
        assert model.get_params() == {
            "C": 0.5,
            "penalty": None,
            "fit_intercept": True,
            "solver": "gd",
            "lr": 0.05,
            "max_iter": 50,
            "tol": 1e-6,
        }

    def test_coef_shape_matches_n_features(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=5, centers=2, random_state=0)
        model = LogisticRegression().fit(X, y)
        assert model.coef_.shape == (5,)

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = LogisticRegression().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_non_binary_target_raises(self) -> None:
        X, y = make_blobs(n_samples=60, centers=3, random_state=0)
        with pytest.raises(ValueError, match="binary"):
            LogisticRegression().fit(X, y)

    def test_unknown_solver_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="solver must be one of"):
            LogisticRegression(solver="lbfgs").fit(X, y)

    def test_unknown_penalty_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="penalty must be one of"):
            LogisticRegression(penalty="l1").fit(X, y)

    def test_non_positive_C_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="C must be > 0"):
            LogisticRegression(C=0.0).fit(X, y)

    def test_max_iter_below_one_raises(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, random_state=0)
        with pytest.raises(ValueError, match="max_iter"):
            LogisticRegression(max_iter=0).fit(X, y)

    def test_predict_proba_rows_sum_to_one(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        proba = LogisticRegression().fit(X, y).predict_proba(X)
        assert proba.shape == (60, 2)
        assert proba.sum(axis=1) == pytest.approx(np.ones(60))

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        model = LogisticRegression().fit(X, y)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, model.predict(X)))

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(LogisticRegression()) == (
            "LogisticRegression(C=1.0, penalty='l2', fit_intercept=True, "
            "solver='newton', lr=0.1, max_iter=1000, tol=1e-06)"
        )


class TestBehavioral:
    def test_separable_blobs_are_classified_almost_perfectly(self) -> None:
        X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
        model = LogisticRegression().fit(X, y)
        assert model.score(X, y) > 0.95

    def test_beats_majority_baseline_on_noisy_moons(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.2, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        model = LogisticRegression().fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.8

    def test_larger_x_raises_probability_when_coef_positive(self) -> None:
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression().fit(X, y)
        proba = model.predict_proba(np.array([[-3.0], [0.0], [3.0]]))[:, 1]
        assert proba[0] < proba[1] < proba[2]


class TestEdgeCases:
    def test_fit_intercept_false(self) -> None:
        X, y = make_blobs(n_samples=80, centers=2, cluster_std=2.0, random_state=4)
        X = X - X.mean(axis=0)
        model = LogisticRegression(fit_intercept=False).fit(X, y)
        assert model.intercept_ == 0.0
        assert model.coef_.shape == (X.shape[1],)

    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=1, centers=2, random_state=5)
        model = LogisticRegression().fit(X, y)
        assert model.coef_.shape == (1,)

    def test_gd_hitting_max_iter_warns_and_records_n_iter(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, cluster_std=2.0, random_state=6)
        model = LogisticRegression(solver="gd", lr=1e-4, max_iter=3, tol=1e-12)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            model.fit(X, y)
        assert model.n_iter_ == 3

    def test_newton_hitting_max_iter_warns(self) -> None:
        X, y = make_blobs(n_samples=30, centers=2, cluster_std=2.0, random_state=7)
        model = LogisticRegression(solver="newton", max_iter=1, tol=1e-14)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            model.fit(X, y)
        assert model.n_iter_ == 1

    def test_predict_labels_are_drawn_from_classes(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, cluster_std=1.0, random_state=8)
        model = LogisticRegression().fit(X, 2 * y - 1)  # labels {-1, 1}
        assert model.classes_.tolist() == [-1.0, 1.0]
        assert set(np.unique(model.predict(X))).issubset({-1.0, 1.0})
