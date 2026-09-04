"""Tests for scratchgrad.linear.LinearRegression.

Tiers (plan.md section 3): analytic closed-form checks, a gradient check on
the GD update, solver-consistency, the fit/predict contract, one behavioral
check on synthetic data, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_regression
from scratchgrad.exceptions import ConvergenceWarning, NotFittedError
from scratchgrad.linear import LinearRegression
from scratchgrad.linear.linear_regression import _mse_gradient, _mse_objective
from scratchgrad.metrics import r2_score
from scratchgrad.preprocessing import StandardScaler
from tests.conftest import ATOL, RTOL
from tests.helpers.gradcheck import gradient_check


class TestAnalytic:
    def test_recovers_a_hand_computed_line(self) -> None:
        # Four points exactly on y = 2x + 1.
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        model = LinearRegression().fit(X, y)
        assert model.coef_ == pytest.approx([2.0], abs=1e-10)
        assert model.intercept_ == pytest.approx(1.0, abs=1e-10)

    def test_normal_solver_matches_numpy_lstsq(self, rng) -> None:
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        model = LinearRegression().fit(X, y)

        X_aug = np.column_stack([np.ones(20), X])
        theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        assert model.intercept_ == pytest.approx(theta[0], rel=RTOL)
        assert model.coef_ == pytest.approx(theta[1:], rel=RTOL)

    def test_noise_free_data_is_fit_exactly(self, rng) -> None:
        X, y = make_regression(n_samples=50, n_features=4, noise=0.0, random_state=0)
        model = LinearRegression().fit(X, y)
        assert r2_score(y, model.predict(X)) == pytest.approx(1.0, abs=ATOL)
        assert model.predict(X) == pytest.approx(y, abs=1e-9)


class TestGradient:
    def test_gd_gradient_matches_finite_differences(self, rng) -> None:
        X_aug = rng.standard_normal((15, 4))
        y = rng.standard_normal(15)
        theta = rng.standard_normal(4)
        analytic = _mse_gradient(X_aug, y, theta)
        gradient_check(lambda t: _mse_objective(X_aug, y, t), analytic, theta)


class TestSolverConsistency:
    def test_gd_converges_to_the_normal_equation(self, rng) -> None:
        X, y = make_regression(n_samples=200, n_features=3, noise=0.2, random_state=1)
        X = StandardScaler().fit_transform(X)  # GD needs scaled features

        exact = LinearRegression(solver="normal").fit(X, y)
        approx = LinearRegression(solver="gd", lr=0.1, max_iter=20_000, tol=1e-10).fit(
            X, y
        )

        assert approx.coef_ == pytest.approx(exact.coef_, abs=1e-5)
        assert approx.intercept_ == pytest.approx(exact.intercept_, abs=1e-5)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            LinearRegression().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self, rng) -> None:
        model = LinearRegression()
        assert model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10)) is model

    def test_fit_does_not_mutate_hyperparameters(self, rng) -> None:
        model = LinearRegression(solver="gd", lr=0.05, max_iter=500, tol=1e-4)
        model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10))
        assert model.get_params() == {
            "fit_intercept": True,
            "solver": "gd",
            "lr": 0.05,
            "max_iter": 500,
            "tol": 1e-4,
        }

    def test_coef_shape_matches_n_features(self, rng) -> None:
        model = LinearRegression().fit(
            rng.standard_normal((10, 5)), rng.standard_normal(10)
        )
        assert model.coef_.shape == (5,)

    def test_predict_rejects_wrong_feature_count(self, rng) -> None:
        model = LinearRegression().fit(
            rng.standard_normal((10, 3)), rng.standard_normal(10)
        )
        with pytest.raises(ValueError, match="features"):
            model.predict(rng.standard_normal((4, 2)))

    def test_score_is_r2(self, rng) -> None:
        X, y = make_regression(n_samples=60, n_features=2, noise=0.1, random_state=2)
        model = LinearRegression().fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))


class TestBehavioral:
    def test_high_score_on_low_noise_data(self, rng) -> None:
        X, y = make_regression(n_samples=300, n_features=5, noise=0.1, random_state=3)
        n_train = 200
        model = LinearRegression().fit(X[:n_train], y[:n_train])
        assert model.score(X[n_train:], y[n_train:]) > 0.99


class TestEdgeCases:
    def test_fit_intercept_false_on_centered_data(self, rng) -> None:
        X, y = make_regression(n_samples=80, n_features=3, noise=0.0, random_state=4)
        X = X - X.mean(axis=0)
        y = y - y.mean()
        model = LinearRegression(fit_intercept=False).fit(X, y)
        assert model.intercept_ == 0.0
        assert model.score(X, y) == pytest.approx(1.0, abs=1e-8)

    def test_single_feature(self, rng) -> None:
        X, y = make_regression(n_samples=40, n_features=1, noise=0.0, random_state=5)
        model = LinearRegression().fit(X, y)
        assert model.coef_.shape == (1,)
        assert model.score(X, y) == pytest.approx(1.0, abs=1e-8)

    def test_gd_hitting_max_iter_warns_and_records_n_iter(self, rng) -> None:
        X, y = make_regression(n_samples=30, n_features=2, noise=0.1, random_state=6)
        model = LinearRegression(solver="gd", lr=1e-3, max_iter=3, tol=1e-12)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            model.fit(X, y)
        assert model.n_iter_ == 3

    def test_unknown_solver_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="solver must be one of"):
            LinearRegression(solver="lstsq").fit(
                rng.standard_normal((5, 2)), rng.standard_normal(5)
            )

    def test_gd_max_iter_below_one_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            LinearRegression(solver="gd", max_iter=0).fit(
                rng.standard_normal((5, 2)), rng.standard_normal(5)
            )

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(LinearRegression()) == (
            "LinearRegression(fit_intercept=True, solver='normal', lr=0.01, "
            "max_iter=1000, tol=1e-06)"
        )
