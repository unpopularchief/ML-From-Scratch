"""Tests for scratchgrad.cluster.GaussianMixture.

Tiers (plan.md section 3): GMM has a real objective (log-likelihood) but no
gradient exposed publicly, so the correctness analogues are (a) EM's
average log-likelihood is non-decreasing every iteration
(docs/derivations/gaussian_mixture.md section 4), (b) a hand-computed
weighted-MLE M-step for each covariance_type, and (c) cross-checking
covariance_type="full" against "diag" on data with a genuinely diagonal
true covariance. Plus the KMeans limiting case, n_init, convergence,
determinism, the fit/predict contract, generative sampling, bic/aic, and
edge cases (including a deliberately singular covariance).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scratchgrad.cluster import GaussianMixture, KMeans
from scratchgrad.cluster.gaussian_mixture import (
    _e_step,
    _init_random_resp,
    _m_step,
    _n_parameters,
)
from scratchgrad.datasets import make_blobs
from scratchgrad.exceptions import NotFittedError


def _best_permutation_agreement(
    true: np.ndarray, pred: np.ndarray, n_classes: int
) -> float:
    """Best agreement fraction over every relabelling (cluster labels are arbitrary)."""
    best = 0.0
    for perm in itertools.permutations(range(n_classes)):
        remapped = np.array(perm)[pred]
        best = max(best, np.mean(remapped == true))
    return best


class TestAnalytic:
    def test_m_step_full_matches_hand_computed_weighted_mle(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        # hard responsibilities: first two points -> component 0, rest -> 1
        resp = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        weights, means, covariances = _m_step(X, resp, "full", reg_covar=0.0)
        np.testing.assert_allclose(weights, [0.5, 0.5])
        np.testing.assert_allclose(means, [[1.0, 0.0], [11.0, 0.0]])
        # each component's covariance = mean of (x-mu)(x-mu)^T over its 2 points
        np.testing.assert_allclose(covariances[0], [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(covariances[1], [[1.0, 0.0], [0.0, 0.0]])

    def test_m_step_diag_matches_full_diagonal_on_axis_aligned_data(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [2.0, 4.0]])
        resp = np.full((4, 1), 1.0)
        _, _, full_cov = _m_step(X, resp, "full", reg_covar=0.0)
        _, _, diag_cov = _m_step(X, resp, "diag", reg_covar=0.0)
        np.testing.assert_allclose(np.diag(full_cov[0]), diag_cov[0])

    def test_m_step_spherical_is_the_mean_diag_variance(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        resp = np.full((4, 1), 1.0)
        _, _, diag_cov = _m_step(X, resp, "diag", reg_covar=0.0)
        _, _, spherical_cov = _m_step(X, resp, "spherical", reg_covar=0.0)
        assert spherical_cov[0] == pytest.approx(diag_cov[0].mean())

    def test_m_step_tied_pools_scatter_across_components(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        resp = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        _, _, full_cov = _m_step(X, resp, "full", reg_covar=0.0)
        _, _, tied_cov = _m_step(X, resp, "tied", reg_covar=0.0)
        # equal-sized components here -> tied covariance is the plain average
        np.testing.assert_allclose(tied_cov, (full_cov[0] + full_cov[1]) / 2.0)

    def test_single_component_recovers_the_global_mean_and_covariance(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        model = GaussianMixture(n_components=1, covariance_type="full").fit(X)
        np.testing.assert_allclose(model.means_[0], X.mean(axis=0))
        expected_cov = np.cov(X.T, bias=True)
        np.testing.assert_allclose(model.covariances_[0], expected_cov, atol=1e-4)

    def test_e_step_responsibilities_sum_to_one(self) -> None:
        X = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
        weights = np.array([0.5, 0.5])
        means = np.array([[0.0, 0.0], [10.0, 10.0]])
        covariances = np.array([np.eye(2), np.eye(2)])
        _, log_resp = _e_step(X, weights, means, covariances, "full")
        np.testing.assert_allclose(np.exp(log_resp).sum(axis=1), 1.0)


class TestMonotonicity:
    def test_average_log_likelihood_never_decreases(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        resp = _init_random_resp(X.shape[0], 4, rng)
        weights, means, covariances = _m_step(X, resp, "full", reg_covar=1e-6)
        lower_bounds = []
        for _ in range(15):
            log_prob_norm, log_resp = _e_step(X, weights, means, covariances, "full")
            lower_bounds.append(float(log_prob_norm.mean()))
            weights, means, covariances = _m_step(
                X, np.exp(log_resp), "full", reg_covar=1e-6
            )
        assert all(a <= b + 1e-9 for a, b in itertools.pairwise(lower_bounds))


class TestKMeansLimit:
    def test_spherical_with_tiny_variance_agrees_with_kmeans(self) -> None:
        X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0)
        gmm = GaussianMixture(
            n_components=3, covariance_type="spherical", reg_covar=1e-8, random_state=0
        ).fit(X)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        # both should recover the same partition, up to a label permutation
        agreement = _best_permutation_agreement(kmeans.labels_, gmm.labels_, 3)
        assert agreement > 0.95


class TestIndependentReference:
    def test_two_points_one_component_closed_form(self) -> None:
        X = np.array([[0.0, 0.0], [4.0, 0.0]])
        model = GaussianMixture(n_components=1, covariance_type="full").fit(X)
        np.testing.assert_allclose(model.means_[0], [2.0, 0.0])
        # variance of {0, 4} (biased) = 4.0 along axis 0, 0 along axis 1
        np.testing.assert_allclose(model.covariances_[0][0, 0], 4.0, atol=1e-4)
        np.testing.assert_allclose(model.covariances_[0][1, 1], 0.0, atol=1e-4)


class TestNInit:
    def test_best_of_n_init_is_geq_a_single_restart(self) -> None:
        X, _ = make_blobs(n_samples=100, centers=4, cluster_std=3.0, random_state=0)
        single = (
            GaussianMixture(
                n_components=4, n_init=1, init_params="random", random_state=0
            )
            .fit(X)
            .lower_bound_
        )
        best_of_many = (
            GaussianMixture(
                n_components=4, n_init=5, init_params="random", random_state=0
            )
            .fit(X)
            .lower_bound_
        )
        assert best_of_many >= single - 1e-9


class TestConvergence:
    def test_n_iter_never_exceeds_max_iter(self) -> None:
        X, _ = make_blobs(n_samples=200, centers=5, random_state=0)
        model = GaussianMixture(n_components=5, max_iter=3, random_state=0).fit(X)
        assert model.n_iter_ <= 3  # noqa: PLR2004

    def test_tiny_max_iter_does_not_converge(self) -> None:
        X, _ = make_blobs(n_samples=200, centers=5, cluster_std=3.0, random_state=0)
        model = GaussianMixture(n_components=5, max_iter=1, random_state=0).fit(X)
        assert model.converged_ is False

    def test_ample_max_iter_converges_on_easy_data(self) -> None:
        X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0)
        model = GaussianMixture(n_components=3, max_iter=200, random_state=0).fit(X)
        assert model.converged_ is True


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        X, _ = make_blobs(n_samples=150, centers=4, random_state=0)
        a = GaussianMixture(n_components=4, random_state=42).fit(X)
        b = GaussianMixture(n_components=4, random_state=42).fit(X)
        np.testing.assert_allclose(a.means_, b.means_)
        np.testing.assert_allclose(a.covariances_, b.covariances_)
        np.testing.assert_array_equal(a.labels_, b.labels_)

    def test_different_seed_can_differ(self) -> None:
        X, _ = make_blobs(n_samples=150, centers=6, cluster_std=5.0, random_state=0)
        a = GaussianMixture(n_components=6, init_params="random", random_state=1).fit(X)
        b = GaussianMixture(n_components=6, init_params="random", random_state=2).fit(X)
        assert not np.allclose(sorted(a.means_.tolist()), sorted(b.means_.tolist()))


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianMixture().predict(np.zeros((2, 2)))

    def test_predict_proba_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianMixture().predict_proba(np.zeros((2, 2)))

    def test_score_samples_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianMixture().score_samples(np.zeros((2, 2)))

    def test_sample_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianMixture().sample(5)

    def test_bic_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            GaussianMixture().bic(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        model = GaussianMixture(n_components=2)
        assert model.fit(X) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        model = GaussianMixture(
            n_components=3,
            covariance_type="diag",
            tol=1e-2,
            reg_covar=1e-5,
            max_iter=50,
            n_init=2,
            init_params="random",
            random_state=7,
        )
        model.fit(X)
        assert model.get_params() == {
            "n_components": 3,
            "covariance_type": "diag",
            "tol": 1e-2,
            "reg_covar": 1e-5,
            "max_iter": 50,
            "n_init": 2,
            "init_params": "random",
            "random_state": 7,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, _ = make_blobs(n_samples=30, n_features=3, centers=2, random_state=0)
        model = GaussianMixture(n_components=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_bad_n_components_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            GaussianMixture(n_components=0).fit(X)

    def test_n_components_larger_than_n_samples_raises(self) -> None:
        X, _ = make_blobs(n_samples=5, centers=2, random_state=0)
        with pytest.raises(ValueError, match="larger than the number of"):
            GaussianMixture(n_components=6).fit(X)

    def test_bad_covariance_type_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="covariance_type must be one of"):
            GaussianMixture(n_components=2, covariance_type="banded").fit(X)

    def test_bad_init_params_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="init_params must be one of"):
            GaussianMixture(n_components=2, init_params="forgy").fit(X)

    def test_bad_n_init_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_init must be >= 1"):
            GaussianMixture(n_components=2, n_init=0).fit(X)

    def test_bad_max_iter_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            GaussianMixture(n_components=2, max_iter=0).fit(X)

    def test_negative_tol_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="tol must be >= 0"):
            GaussianMixture(n_components=2, tol=-1.0).fit(X)

    def test_negative_reg_covar_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="reg_covar must be >= 0"):
            GaussianMixture(n_components=2, reg_covar=-1.0).fit(X)

    def test_sample_rejects_non_positive_n_samples(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        model = GaussianMixture(n_components=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            model.sample(0)

    def test_predict_proba_rows_sum_to_one(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = GaussianMixture(n_components=3, random_state=0).fit(X)
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)

    def test_predict_matches_labels(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = GaussianMixture(n_components=3, random_state=0).fit(X)
        np.testing.assert_array_equal(model.predict(X), model.labels_)

    def test_fit_predict_matches_fit_then_labels(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        a = GaussianMixture(n_components=3, random_state=0).fit(X).labels_
        b = GaussianMixture(n_components=3, random_state=0).fit_predict(X)
        np.testing.assert_array_equal(a, b)

    def test_score_is_mean_of_score_samples(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = GaussianMixture(n_components=3, random_state=0).fit(X)
        assert model.score(X) == pytest.approx(model.score_samples(X).mean())

    def test_repr_round_trips(self) -> None:
        assert repr(GaussianMixture()) == (
            "GaussianMixture(n_components=1, covariance_type='full', tol=0.001, "
            "reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', "
            "random_state=None)"
        )


class TestBehavioral:
    def test_recovers_well_separated_blobs(self) -> None:
        # n_init=1 (the default, matching sklearn) is as local-optimum
        # sensitive as KMeans's own n_init=1 -- a few restarts is the
        # realistic-usage defense (docs/derivations/gaussian_mixture.md §7).
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)
        model = GaussianMixture(n_components=4, n_init=5, random_state=0).fit(X)
        assert _best_permutation_agreement(y, model.labels_, 4) > 0.95


class TestGenerative:
    def test_sample_mean_and_covariance_converge(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.multivariate_normal([1.0, -2.0], [[2.0, 0.5], [0.5, 1.0]], size=300)
        model = GaussianMixture(n_components=1, covariance_type="full").fit(X)
        samples, labels = model.sample(20000)
        assert samples.shape == (20000, 2)
        np.testing.assert_array_equal(labels, 0)
        np.testing.assert_allclose(samples.mean(axis=0), model.means_[0], atol=0.1)
        np.testing.assert_allclose(
            np.cov(samples.T, bias=True), model.covariances_[0], atol=0.15
        )

    @pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
    def test_sample_works_for_every_covariance_type(self, covariance_type: str) -> None:
        X, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=0)
        model = GaussianMixture(
            n_components=3, covariance_type=covariance_type, random_state=0
        ).fit(X)
        samples, labels = model.sample(50)
        assert samples.shape == (50, 2)
        assert set(labels.tolist()) <= {0, 1, 2}

    def test_sample_skips_a_component_with_zero_draws(self) -> None:
        # a component with (near) zero weight is very unlikely to be drawn
        # at all in a small sample -- exercise that skip explicitly.
        X, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=0)
        model = GaussianMixture(n_components=3, random_state=0).fit(X)
        model.weights_ = np.array([1.0, 0.0, 0.0])
        samples, labels = model.sample(5)
        assert samples.shape == (5, 2)
        np.testing.assert_array_equal(labels, 0)

    def test_sample_component_frequencies_track_weights(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal([0.0, 0.0], 0.5, size=(150, 2)),
                rng.normal([20.0, 20.0], 0.5, size=(100, 2)),
                rng.normal([40.0, 0.0], 0.5, size=(50, 2)),
            ]
        )
        model = GaussianMixture(n_components=3, random_state=0).fit(X)
        _, labels = model.sample(6000)
        empirical = np.bincount(labels, minlength=3) / 6000
        np.testing.assert_allclose(empirical, model.weights_, atol=0.05)


class TestBicAic:
    @pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
    def test_bic_matches_hand_computed_value(self, covariance_type: str) -> None:
        X, _ = make_blobs(n_samples=60, n_features=2, centers=2, random_state=0)
        model = GaussianMixture(
            n_components=2, covariance_type=covariance_type, random_state=0
        ).fit(X)
        n_params = _n_parameters(2, 2, covariance_type)
        expected_bic = -2 * X.shape[0] * model.score(X) + n_params * np.log(X.shape[0])
        expected_aic = -2 * X.shape[0] * model.score(X) + 2 * n_params
        assert model.bic(X) == pytest.approx(expected_bic)
        assert model.aic(X) == pytest.approx(expected_aic)

    def test_more_components_can_reduce_bic_on_genuinely_multimodal_data(self) -> None:
        X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)
        bic_1 = GaussianMixture(n_components=1, random_state=0).fit(X).bic(X)
        bic_4 = GaussianMixture(n_components=4, random_state=0).fit(X).bic(X)
        assert bic_4 < bic_1


class TestEdgeCases:
    def test_n_components_equals_one(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = GaussianMixture(n_components=1, random_state=0).fit(X)
        np.testing.assert_allclose(model.means_[0], X.mean(axis=0))

    def test_n_components_equals_n_samples(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=3, random_state=0)
        model = GaussianMixture(n_components=10, reg_covar=1e-3, random_state=0).fit(X)
        assert model.means_.shape == (10, 2)

    def test_singular_covariance_without_reg_covar_raises_clear_error(self) -> None:
        # 3 distinct points, 3 components, reg_covar=0: the KMeans init puts
        # exactly one point per component, so every "full" covariance is an
        # exact zero matrix -- not positive-definite.
        X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        with pytest.raises(ValueError, match="reg_covar"):
            GaussianMixture(n_components=3, reg_covar=0.0, random_state=0).fit(X)

    def test_single_feature(self) -> None:
        X, _ = make_blobs(n_samples=40, n_features=1, centers=2, random_state=0)
        model = GaussianMixture(n_components=2, random_state=0).fit(X)
        assert model.means_.shape == (2, 1)
