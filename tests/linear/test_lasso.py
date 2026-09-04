"""Tests for scratchgrad.linear.Lasso.

Tiers (plan.md section 3): analytic checks on the soft-threshold and a
single-feature fit, a KKT/optimality check standing in for the (impossible)
gradient check, a reduction to OLS at alpha=0, sparsity and monotonicity
checks, objective monotonicity, the fit/predict contract, one behavioral
check on sparse data, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_regression
from scratchgrad.exceptions import ConvergenceWarning, NotFittedError
from scratchgrad.linear import Lasso, LinearRegression
from scratchgrad.linear.lasso import _lasso_objective, _soft_threshold
from scratchgrad.metrics import r2_score
from tests.conftest import RTOL


class TestSoftThreshold:
    def test_matches_definition(self) -> None:
        assert _soft_threshold(5.0, 2.0) == pytest.approx(3.0)  # rho > alpha
        assert _soft_threshold(-5.0, 2.0) == pytest.approx(-3.0)  # rho < -alpha
        assert _soft_threshold(1.5, 2.0) == 0.0  # |rho| < alpha
        assert _soft_threshold(-1.5, 2.0) == 0.0
        assert _soft_threshold(2.0, 2.0) == 0.0  # boundary
        assert _soft_threshold(3.0, 0.0) == pytest.approx(3.0)  # no penalty


class TestAnalytic:
    def test_single_standardised_feature_is_soft_thresholded(self, rng) -> None:
        n = 200
        x = rng.standard_normal(n)
        x = (x - x.mean()) / np.sqrt(np.mean((x - x.mean()) ** 2))  # mean 0, z = 1
        y = 2.0 * x + 0.1 * rng.standard_normal(n)
        X = x.reshape(-1, 1)

        for alpha in (0.5, 1.0):
            rho = float(X[:, 0] @ y / n)  # z = 1, so w = S(rho, alpha)
            model = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12).fit(X, y)
            assert model.coef_[0] == pytest.approx(
                _soft_threshold(rho, alpha), abs=1e-9
            )

    def test_large_alpha_zeros_the_only_feature(self, rng) -> None:
        n = 200
        x = rng.standard_normal(n)
        y = 2.0 * x + 0.1 * rng.standard_normal(n)
        model = Lasso(alpha=50.0, fit_intercept=False).fit(x.reshape(-1, 1), y)
        assert model.coef_[0] == 0.0


class TestOptimality:
    def test_kkt_conditions_hold_at_the_solution(self, rng) -> None:
        X, y = make_regression(n_samples=150, n_features=8, noise=0.3, random_state=0)
        alpha = 0.2
        model = Lasso(alpha=alpha, tol=1e-10, max_iter=100_000).fit(X, y)

        n = X.shape[0]
        X_c = X - X.mean(axis=0)
        y_c = y - y.mean()
        w = model.coef_
        # c_j = (1/n) x_j^T (y_c - X_c w) — the negative gradient of the
        # smooth part. KKT: c_j = alpha*sign(w_j) where w_j != 0, |c_j| <= alpha
        # where w_j == 0.
        c = X_c.T @ (y_c - X_c @ w) / n
        nonzero = w != 0.0
        assert np.allclose(c[nonzero], alpha * np.sign(w[nonzero]), atol=1e-6)
        assert np.all(np.abs(c[~nonzero]) <= alpha + 1e-6)


class TestReductionToOLS:
    def test_alpha_zero_matches_linear_regression(self, rng) -> None:
        X, y = make_regression(n_samples=80, n_features=4, noise=0.3, random_state=0)
        lasso = Lasso(alpha=0.0, tol=1e-12, max_iter=100_000).fit(X, y)
        ols = LinearRegression().fit(X, y)
        assert lasso.coef_ == pytest.approx(ols.coef_, rel=1e-4, abs=1e-6)
        assert lasso.intercept_ == pytest.approx(ols.intercept_, rel=1e-4, abs=1e-6)


class TestSparsity:
    def test_large_alpha_produces_exact_zeros(self, rng) -> None:
        X, y = make_regression(n_samples=120, n_features=10, noise=0.2, random_state=1)
        model = Lasso(alpha=1.0).fit(X, y)
        assert np.any(model.coef_ == 0.0)

    def test_huge_alpha_zeros_everything_and_intercept_is_the_mean(self, rng) -> None:
        X, y = make_regression(n_samples=60, n_features=5, noise=0.1, random_state=2)
        model = Lasso(alpha=1e6).fit(X, y)
        assert np.all(model.coef_ == 0.0)
        assert model.intercept_ == pytest.approx(y.mean())

    def test_nonzero_count_and_l1_norm_shrink_as_alpha_grows(self, rng) -> None:
        X, y = make_regression(n_samples=150, n_features=10, noise=0.2, random_state=3)
        alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]
        fits = [Lasso(alpha=a, max_iter=50_000).fit(X, y).coef_ for a in alphas]
        n_nonzero = [int(np.sum(w != 0.0)) for w in fits]
        l1 = [float(np.sum(np.abs(w))) for w in fits]
        assert all(b <= a for a, b in zip(n_nonzero[:-1], n_nonzero[1:], strict=True))
        assert all(b <= a for a, b in zip(l1[:-1], l1[1:], strict=True))


class TestObjectiveDecreases:
    @pytest.mark.filterwarnings("ignore:Coordinate descent did not converge")
    def test_more_sweeps_never_increase_the_objective(self, rng) -> None:
        X, y = make_regression(n_samples=100, n_features=6, noise=0.3, random_state=4)
        alpha = 0.3
        objectives = []
        for max_iter in (1, 2, 5, 20, 100):
            model = Lasso(alpha=alpha, max_iter=max_iter, tol=0.0).fit(X, y)
            X_c = X - X.mean(axis=0)
            y_c = y - y.mean()
            objectives.append(_lasso_objective(X_c, y_c, model.coef_, 0.0, alpha))
        assert all(
            b <= a + 1e-12 for a, b in zip(objectives[:-1], objectives[1:], strict=True)
        )


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            Lasso().predict(np.zeros((2, 2)))

    def test_fit_returns_self(self, rng) -> None:
        model = Lasso()
        assert model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10)) is model

    def test_fit_does_not_mutate_hyperparameters(self, rng) -> None:
        model = Lasso(alpha=0.5, fit_intercept=False, max_iter=250, tol=1e-3)
        model.fit(rng.standard_normal((10, 2)), rng.standard_normal(10))
        assert model.get_params() == {
            "alpha": 0.5,
            "fit_intercept": False,
            "max_iter": 250,
            "tol": 1e-3,
        }

    def test_coef_shape_matches_n_features(self, rng) -> None:
        model = Lasso().fit(rng.standard_normal((10, 5)), rng.standard_normal(10))
        assert model.coef_.shape == (5,)

    def test_predict_rejects_wrong_feature_count(self, rng) -> None:
        model = Lasso().fit(rng.standard_normal((10, 3)), rng.standard_normal(10))
        with pytest.raises(ValueError, match="features"):
            model.predict(rng.standard_normal((4, 2)))

    def test_score_is_r2(self, rng) -> None:
        X, y = make_regression(n_samples=60, n_features=2, noise=0.1, random_state=2)
        model = Lasso(alpha=0.01).fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))


class TestBehavioral:
    def test_recovers_a_sparse_signal_and_beats_ols(self, rng) -> None:
        n, d = 120, 60  # more features than training samples -> OLS overfits
        X = rng.standard_normal((n, d))
        true_w = np.zeros(d)
        true_w[:3] = [3.0, -2.0, 1.5]  # only 3 of 60 features matter
        y = X @ true_w + 0.5 * rng.standard_normal(n)

        n_train = 70
        lasso = Lasso(alpha=0.1, max_iter=50_000).fit(X[:n_train], y[:n_train])
        ols = LinearRegression().fit(X[:n_train], y[:n_train])

        assert np.sum(lasso.coef_ == 0.0) >= 45  # zeros out most irrelevant features
        assert lasso.score(X[n_train:], y[n_train:]) > ols.score(
            X[n_train:], y[n_train:]
        )


class TestEdgeCases:
    def test_fit_intercept_false_on_centered_data(self, rng) -> None:
        X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=4)
        X = X - X.mean(axis=0)
        y = y - y.mean()
        with_flag = Lasso(alpha=0.1, fit_intercept=False).fit(X, y)
        centered = Lasso(alpha=0.1, fit_intercept=True).fit(X, y)
        assert with_flag.intercept_ == 0.0
        assert with_flag.coef_ == pytest.approx(centered.coef_, rel=RTOL, abs=1e-8)

    def test_single_feature(self, rng) -> None:
        X, y = make_regression(n_samples=40, n_features=1, noise=0.1, random_state=5)
        model = Lasso(alpha=0.05).fit(X, y)
        assert model.coef_.shape == (1,)

    def test_constant_feature_gets_zero_weight(self, rng) -> None:
        n = 60
        X = np.column_stack([rng.standard_normal(n), np.full(n, 3.0)])
        y = rng.standard_normal(n)
        model = Lasso(alpha=0.01).fit(X, y)
        assert model.coef_[1] == 0.0

    def test_max_iter_hit_warns_and_records_n_iter(self, rng) -> None:
        X, y = make_regression(n_samples=60, n_features=5, noise=0.2, random_state=6)
        model = Lasso(alpha=0.01, max_iter=1, tol=1e-12)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            model.fit(X, y)
        assert model.n_iter_ == 1

    def test_negative_alpha_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            Lasso(alpha=-1.0).fit(rng.standard_normal((5, 2)), rng.standard_normal(5))

    def test_max_iter_below_one_raises(self, rng) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            Lasso(max_iter=0).fit(rng.standard_normal((5, 2)), rng.standard_normal(5))

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(Lasso()) == (
            "Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.0001)"
        )
