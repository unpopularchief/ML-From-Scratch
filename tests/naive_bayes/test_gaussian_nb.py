"""Tests for scratchgrad.naive_bayes.GaussianNB.

Tiers (plan.md section 3): analytic hand-computed means/vars/priors and a
by-hand joint log-likelihood; an independent Python recomputation of the
log-posterior (the correctness analogue of a gradient check — GaussianNB has
no gradient); the var_smoothing floor; proba/log-proba consistency; the
priors knob; the fit/predict contract; a behavioral check on correlated
features; and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs
from scratchgrad.exceptions import NotFittedError
from scratchgrad.naive_bayes import GaussianNB
from scratchgrad.naive_bayes.gaussian_nb import (
    _gaussian_log_density,
    _joint_log_likelihood,
)
from scratchgrad.preprocessing import train_test_split


def _brute_jll(
    X: np.ndarray, mean: np.ndarray, var: np.ndarray, log_prior: np.ndarray
) -> np.ndarray:
    """Dead-simple triple loop over samples, classes, features."""
    n, n_classes = X.shape[0], mean.shape[0]
    out = np.zeros((n, n_classes))
    for i in range(n):
        for c in range(n_classes):
            total = log_prior[c]
            for j in range(X.shape[1]):
                m, v = mean[c, j], var[c, j]
                total += -0.5 * np.log(2.0 * np.pi * v) - (X[i, j] - m) ** 2 / (2.0 * v)
            out[i, c] = total
    return out


class TestGaussianLogDensity:
    def test_matches_the_formula_on_a_hand_case(self) -> None:
        X = np.array([[0.0, 0.0]])
        mean = np.array([[0.0, 1.0], [2.0, 2.0]])
        var = np.array([[1.0, 1.0], [0.5, 4.0]])
        got = _gaussian_log_density(X, mean, var)
        expected = np.zeros((1, 2, 2))
        for c in range(2):
            for j in range(2):
                v = var[c, j]
                expected[0, c, j] = -0.5 * (
                    np.log(2.0 * np.pi * v) + (0.0 - mean[c, j]) ** 2 / v
                )
        assert got.shape == (1, 2, 2)
        np.testing.assert_allclose(got, expected)


class TestAnalytic:
    def test_closed_form_mean_var_prior(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [10.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1])
        model = GaussianNB(var_smoothing=0.0).fit(X, y)

        assert model.epsilon_ == 0.0
        np.testing.assert_array_equal(model.classes_, [0.0, 1.0])
        np.testing.assert_array_equal(model.class_count_, [3.0, 2.0])
        np.testing.assert_allclose(model.class_prior_, [0.6, 0.4])
        np.testing.assert_allclose(model.mean_, [[2.0], [11.0]])
        # biased MLE variance: divide by N_c, not N_c - 1
        np.testing.assert_allclose(model.var_, [[2.0 / 3.0], [1.0]])

    def test_joint_log_likelihood_by_hand(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [10.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1])
        model = GaussianNB(var_smoothing=0.0).fit(X, y)

        x = np.array([[2.0]])
        jll = model._joint_log_likelihood(x)[0]
        expected_c0 = np.log(0.6) - 0.5 * np.log(2.0 * np.pi * (2.0 / 3.0))
        expected_c1 = np.log(0.4) - 0.5 * (
            np.log(2.0 * np.pi * 1.0) + (2.0 - 11.0) ** 2 / 1.0
        )
        np.testing.assert_allclose(jll, [expected_c0, expected_c1])
        assert model.predict(x)[0] == 0.0


class TestIndependentRecomputation:
    def test_jll_predict_proba_match_a_python_loop(self, rng) -> None:
        X = rng.standard_normal((60, 4))
        y = rng.integers(0, 3, size=60).astype(np.float64)
        X_query = rng.standard_normal((20, 4))

        model = GaussianNB().fit(X, y)
        log_prior = np.log(model.class_prior_)
        expected = _brute_jll(X_query, model.mean_, model.var_, log_prior)
        np.testing.assert_allclose(model._joint_log_likelihood(X_query), expected)
        # the module-level helper, targeted directly
        np.testing.assert_allclose(
            _joint_log_likelihood(X_query, model.mean_, model.var_, log_prior),
            expected,
        )

        # predict is the argmax of that; predict_proba is its softmax
        np.testing.assert_array_equal(
            model.predict(X_query), model.classes_[np.argmax(expected, axis=1)]
        )
        row_norm = expected - np.log(np.sum(np.exp(expected), axis=1, keepdims=True))
        np.testing.assert_allclose(model.predict_proba(X_query), np.exp(row_norm))


class TestVarSmoothing:
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [[0.0, 5.0], [0.0, 6.0], [0.0, 7.0], [1.0, 20.0], [2.0, 22.0], [3.0, 24.0]]
        )
        y = np.array([0, 0, 0, 1, 1, 1])  # feature 0 is constant within class 0
        return X, y

    def test_constant_in_class_feature_stays_finite(self) -> None:
        X, y = self._data()
        model = GaussianNB().fit(X, y)
        assert model.var_[0, 0] > 0.0  # would be exactly 0 without the floor
        assert np.isfinite(model.predict_proba(X)).all()

    def test_epsilon_is_a_fraction_of_the_largest_feature_variance(self) -> None:
        X, y = self._data()
        model = GaussianNB(var_smoothing=1e-6).fit(X, y)
        assert model.epsilon_ == pytest.approx(1e-6 * X.var(axis=0).max())

    def test_huge_smoothing_pulls_the_posterior_to_the_prior(self) -> None:
        X, y = self._data()
        model = GaussianNB(var_smoothing=1e6).fit(X, y)
        expected = np.tile(model.class_prior_, (X.shape[0], 1))
        np.testing.assert_allclose(model.predict_proba(X), expected, atol=1e-3)


class TestProba:
    def test_shape_rows_sum_to_one_and_agree_with_predict(self) -> None:
        X, y = make_blobs(n_samples=90, centers=3, cluster_std=2.0, random_state=0)
        model = GaussianNB().fit(X, y)
        proba = model.predict_proba(X[:12])
        assert proba.shape == (12, 3)
        assert proba.sum(axis=1) == pytest.approx(np.ones(12))
        labels = model.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(model.predict(X[:12]), labels)

    def test_predict_log_proba_is_log_of_predict_proba(self) -> None:
        X, y = make_blobs(n_samples=80, centers=3, cluster_std=2.0, random_state=1)
        model = GaussianNB().fit(X, y)
        np.testing.assert_allclose(
            model.predict_log_proba(X[:10]), np.log(model.predict_proba(X[:10]))
        )


class TestMulticlass:
    def test_high_accuracy_on_well_separated_multiclass(self) -> None:
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        model = GaussianNB().fit(X_train, y_train)
        assert model.score(X_test, y_test) > 0.9


class TestPriors:
    def test_explicit_priors_override_the_frequencies(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=2, random_state=0)
        y = np.array([0.0] * 30 + [1.0] * 10)  # 3:1 imbalance
        freq = GaussianNB().fit(X, y)
        uniform = GaussianNB(priors=[0.5, 0.5]).fit(X, y)
        np.testing.assert_allclose(freq.class_prior_, [0.75, 0.25])
        np.testing.assert_allclose(uniform.class_prior_, [0.5, 0.5])

    def test_a_skewed_prior_flips_a_borderline_prediction(self) -> None:
        X = np.array([[-2.0], [-1.0], [0.0], [0.0], [1.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])  # symmetric about x = 0
        query = np.array([[0.0]])
        assert GaussianNB().fit(X, y).predict(query)[0] == 0.0  # tie -> lowest label
        assert GaussianNB(priors=[0.1, 0.9]).fit(X, y).predict(query)[0] == 1.0

    def test_bad_priors_raise_at_fit(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="shape"):
            GaussianNB(priors=[0.2, 0.3, 0.5]).fit(X, y)
        with pytest.raises(ValueError, match="sum to 1"):
            GaussianNB(priors=[0.2, 0.3]).fit(X, y)
        with pytest.raises(ValueError, match="non-negative"):
            GaussianNB(priors=[-0.1, 1.1]).fit(X, y)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianNB().predict(np.zeros((2, 2)))

    def test_predict_proba_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianNB().predict_proba(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = GaussianNB()
        assert model.fit(X, y) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, y = make_blobs(n_samples=40, centers=2, random_state=0)
        model = GaussianNB(priors=[0.5, 0.5], var_smoothing=1e-8)
        model.fit(X, y)
        assert model.get_params() == {
            "priors": [0.5, 0.5],
            "var_smoothing": 1e-8,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
        model = GaussianNB().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_negative_var_smoothing_raises(self) -> None:
        X, y = make_blobs(n_samples=20, centers=2, random_state=0)
        with pytest.raises(ValueError, match="var_smoothing must be >= 0"):
            GaussianNB(var_smoothing=-1e-9).fit(X, y)

    def test_score_is_accuracy(self) -> None:
        X, y = make_blobs(n_samples=60, centers=2, cluster_std=2.0, random_state=0)
        model = GaussianNB().fit(X, y)
        preds = model.predict(X)
        assert model.score(X, y) == pytest.approx(np.mean(preds == y))

    def test_repr_round_trips_through_params(self) -> None:
        assert repr(GaussianNB()) == "GaussianNB(priors=None, var_smoothing=1e-09)"


class TestBehavioral:
    def test_near_perfect_on_well_separated_blobs(self) -> None:
        X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=2)
        model = GaussianNB().fit(X, y)
        assert model.score(X, y) > 0.98

    def test_correlated_features_make_predict_proba_overconfident(self, rng) -> None:
        # x1 and x2 are near-duplicates of a latent z; the naive independence
        # assumption then double-counts the same evidence.
        z = rng.standard_normal(500)
        y = (z > 0.0).astype(np.float64)
        x1 = z + 0.6 * rng.standard_normal(500)
        x2 = z + 0.6 * rng.standard_normal(500)
        X = np.column_stack([x1, x2])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        model = GaussianNB().fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        confidence = model.predict_proba(X_test).max(axis=1).mean()
        assert accuracy < 0.95  # the classes genuinely overlap
        assert confidence > accuracy  # ... but the model does not think so


class TestEdgeCases:
    def test_single_feature(self) -> None:
        X, y = make_blobs(n_samples=40, n_features=1, centers=2, random_state=5)
        model = GaussianNB().fit(X, y)
        assert model.predict(X).shape == (40,)
        assert model.mean_.shape == (2, 1)

    def test_one_sample_per_class(self) -> None:
        X = np.array([[0.0, 1.0], [5.0, 6.0]])
        y = np.array([0, 1])
        model = GaussianNB().fit(X, y)  # within-class var 0, floor keeps it finite
        assert model.epsilon_ > 0.0
        assert np.isfinite(model.predict_proba(X)).all()
        np.testing.assert_array_equal(model.predict(X), [0.0, 1.0])

    def test_binary_labels(self, rng) -> None:
        X = rng.standard_normal((50, 3))
        y = rng.integers(0, 2, size=50).astype(np.float64)
        model = GaussianNB().fit(X, y)
        assert set(model.predict(X).tolist()) <= {0.0, 1.0}
