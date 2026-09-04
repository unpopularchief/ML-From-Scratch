"""Tests for scratchgrad.linear.Ridge.

Tiers (plan.md section 3): analytic closed-form checks, a reduction to OLS at
alpha=0, a shrinkage check, a gradient check on the GD update,
solver-consistency, the fit/predict contract, one behavioral check on
collinear data, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_regression
from scratchgrad.exceptions import ConvergenceWarning, NotFittedError
from scratchgrad.linear import LinearRegression, Ridge
from scratchgrad.linear.ridge import _ridge_gradient, _ridge_objective
from scratchgrad.metrics import r2_score
from scratchgrad.preprocessing import StandardScaler
from tests.conftest import RTOL
from tests.helpers.gradcheck import gradient_check


class TestAnalytic:
    def test_hand_solved_system_with_intercept(self) -> None:
        # X = [[1],[2],[3]], y = [2,4,6]. Augmented normal equation with
        # alpha=2 and D=diag(0,1): [[3,6],[6,16]] theta = [12,28] -> b=2, w=1.
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = Ridge(alpha=2.0).fit(X, y)
        assert model.coef_ == pytest.approx([1.0], abs=1e-10)
        assert model.intercept_ == pytest.approx(2.0, abs=1e-10)

    def test_hand_solved_system_without_intercept(self) -> None:
        # Same data, fit_intercept=False: (X^T X + alpha) w = X^T y with
        # X^T X = 14, X^T y = 28, alpha = 14  ->  w = 28 / 28 = 1.
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = Ridge(alpha=14.0, fit_intercept=False).fit(X, y)
        assert model.intercept_ == 0.0
        assert model.coef_ == pytest.approx([1.0], abs=1e-10)

    def test_normal_solver_matches_direct_linear_solve(self, rng) -> None:
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        alpha = 0.7
        model = Ridge(alpha=alpha).fit(X, y)

        X_aug = np.column_stack([np.ones(20), X])
        D = np.diag([0.0, 1.0, 1.0, 1.0])
        theta = np.linalg.solve(X_aug.T @ X_aug + alpha * D, X_aug.T @ y)
        assert model.intercept_ == pytest.approx(theta[0], rel=RTOL)
        assert model.coef_ == pytest.approx(theta[1:], rel=RTOL)


class TestReductionToOLS:
    def test_alpha_zero_matches_linear_regression(self, rng) -> None:
        X, y = make_regression(n_samples=80, n_features=4, noise=0.3, random_state=0)
        ridge = Ridge(alpha=0.0).fit(X, y)
        ols = LinearRegression().fit(X, y)
        assert ridge.coef_ == pytest.approx(ols.coef_, rel=RTOL)
        assert ridge.intercept_ == pytest.approx(ols.intercept_, rel=RTOL)


class TestShrinkage:
    def test_coef_norm_shrinks_monotonically_as_alpha_grows(self, rng) -> None:
        X, y = make_regression(n_samples=120, n_features=5, noise=0.2, random_state=1)
        alphas = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        norms = [np.linalg.norm(Ridge(alpha=a).fit(X, y).coef_) for a in alphas]
        assert all(nxt <= cur for cur, nxt in zip(norms[:-1], norms[1:], strict=True))

    def test_large_alpha_drives_weights_to_zero_and_intercept_to_mean(
        self, rng
    ) -> None:
        X, y = make_regression(n_samples=60, n_features=3, noise=0.1, random_state=2)
        model = Ridge(alpha=1e12).fit(X, y)
        assert model.coef_ == pytest.approx(np.zeros(3), abs=1e-6)
        assert model.intercept_ == pytest.approx(y.mean(), abs=1e-4)


class TestGradient:
    def test_gd_gradient_matches_finite_differences(self, rng) -> None:
        X_aug = rng.standard_normal((15, 4))
        y = rng.standard_normal(15)
        theta = rng.standard_normal(4)
        alpha = 0.7
        penalty_mask = np.array([0.0, 1.0, 1.0, 1.0])  # intercept not penalised
        analytic = _ridge_gradient(X_aug, y, theta, alpha, penalty_mask)
        gradient_check(
            lambda t: _ridge_objective(X_aug, y, t, alpha, penalty_mask),
            analytic,
            theta,
        )


class TestSolverConsistency:
    def test_gd_converges_to_the_normal_equation(self, rng) -> None:
        X, y = make_regression(n_samples=200, n_features=3, noise=0.2, random_state=1)
        X = StandardScaler().fit_transform(X)  # GD needs scaled features

        exact = Ridge(alpha=1.0, solver="normal").fit(X, y)
        approx = Ridge(alpha=1.0, solver="gd", lr=0.1, max_iter=20_000, tol=1e-10).fit(
            X, y
        )

        assert approx.coef_ == pytest.approx(exact.coef_, abs=1e-5)
        assert approx.intercept_ == pytest.approx(exact.intercept_, abs=1e-5)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            Ridge().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self, rng) -> None:
        model = Ridge()
        assert model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10)) is model

    def test_fit_does_not_mutate_hyperparameters(self, rng) -> None:
        model = Ridge(alpha=2.5, solver="gd", lr=0.05, max_iter=500, tol=1e-4)
        model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10))
        assert model.get_params() == {
            "alpha": 2.5,
            "fit_intercept": True,
            "solver": "gd",
            "lr": 0.05,
            "max_iter": 500,
            "tol": 1e-4,
        }

    def test_coef_shape_matches_n_features(self, rng) -> None:
        model = Ridge().fit(rng.standard_normal((10, 5)), rng.standard_normal(10))
        assert model.coef_.shape == (5,)

    def test_predict_rejects_wrong_feature_count(self, rng) -> None:
        model = Ridge().fit(rng.standard_normal((10, 3)), rng.standard_normal(10))
        with pytest.raises(ValueError, match="features"):
            model.predict(rng.standard_normal((4, 2)))

    def test_score_is_r2(self, rng) -> None:
        X, y = make_regression(n_samples=60, n_features=2, noise=0.1, random_state=2)
        model = Ridge().fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))


class TestBehavioral:
    def test_beats_ols_on_collinear_data(self, rng) -> None:
        n = 200
        x1 = rng.standard_normal(n)
        x2 = x1 + 0.01 * rng.standard_normal(n)  # nearly a duplicate of x1
        X = np.column_stack([x1, x2, rng.standard_normal((n, 3))])
        true_w = np.array([1.0, 1.0, 0.5, -0.5, 2.0])
        y = X @ true_w + 0.5 * rng.standard_normal(n)

        n_train = 150
        ols = LinearRegression().fit(X[:n_train], y[:n_train])
        ridge = Ridge(alpha=1.0).fit(X[:n_train], y[:n_train])
        assert ridge.score(X[n_train:], y[n_train:]) > ols.score(
            X[n_train:], y[n_train:]
        )


class TestEdgeCases:
    def test_fit_intercept_false_on_centered_data(self, rng) -> None:
        X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=4)
        X = X - X.mean(axis=0)
        y = y - y.mean()
        model = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
        assert model.intercept_ == 0.0

        X_ridge = np.linalg.solve(
            X.T @ X + 1.0 * np.eye(3), X.T @ y
        )  # no D adjustment: every column is a penalised weight
        assert model.coef_ == pytest.approx(X_ridge, rel=RTOL)

    def test_single_feature(self, rng) -> None:
        X, y = make_regression(n_samples=40, n_features=1, noise=0.1, random_state=5)
        model = Ridge(alpha=0.5).fit(X, y)
        assert model.coef_.shape == (1,)

    def test_gd_hitting_max_iter_warns_and_records_n_iter(self, rng) -> None:
        X, y = make_regression(n_samples=30, n_features=2, noise=0.1, random_state=6)
        model = Ridge(alpha=1.0, solver="gd", lr=1e-3, max_iter=3, tol=1e-12)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            model.fit(X, y)
        assert model.n_iter_ == 3

    def test_unknown_solver_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="solver must be one of"):
            Ridge(solver="cholesky").fit(
                rng.standard_normal((5, 2)), rng.standard_normal(5)
            )

    def test_negative_alpha_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            Ridge(alpha=-1.0).fit(rng.standard_normal((5, 2)), rng.standard_normal(5))

    def test_gd_max_iter_below_one_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            Ridge(solver="gd", max_iter=0).fit(
                rng.standard_normal((5, 2)), rng.standard_normal(5)
            )

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(Ridge()) == (
            "Ridge(alpha=1.0, fit_intercept=True, solver='normal', lr=0.01, "
            "max_iter=1000, tol=1e-06)"
        )
