"""Tests for scratchgrad.decomposition.PCA.

Tiers (plan.md section 3): PCA has no gradient (it's a closed-form
eigenproblem, not an iterative fit), so the correctness analogues are (a)
hand-computed eigendecompositions on tiny constructed data, (b) the
variance-maximisation / reconstruction-error objective equivalence from
docs/derivations/pca.md section 2, and (c) an independently-derived
algorithm (power iteration, section 6) agreeing with the SVD-based `fit`.
Plus the fit/transform contract, determinism, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.decomposition import PCA
from scratchgrad.decomposition.pca import _fix_signs, _power_iteration_pca
from scratchgrad.exceptions import NotFittedError


def _align_signs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flip rows of `a` so each best matches the sign of the same row of `b`."""
    aligned = a.copy()
    for i in range(a.shape[0]):
        if np.dot(aligned[i], b[i]) < 0:
            aligned[i] *= -1.0
    return aligned


class TestAnalytic:
    def test_recovers_the_obvious_axis_of_a_line(self) -> None:
        # points exactly on the y=x line -- all variance lies along
        # [1/sqrt2, 1/sqrt2], none along the orthogonal direction.
        X = np.array([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
        model = PCA(n_components=1).fit(X)
        expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
        component = model.components_[0]
        if component[0] < 0:
            component = -component
        np.testing.assert_allclose(component, expected, atol=1e-10)
        # projections onto the unit diagonal: (x+y)/sqrt(2) for each point
        # -> [-2sqrt2, -sqrt2, sqrt2, 2sqrt2]; sample variance = 20/3.
        np.testing.assert_allclose(model.explained_variance_[0], 20.0 / 3.0)
        np.testing.assert_allclose(model.explained_variance_ratio_[0], 1.0)

    def test_recovers_coordinate_axes_for_diagonal_covariance(self) -> None:
        # independent per-axis spread with no cross-correlation -> the
        # covariance matrix is exactly diagonal, so components == axes.
        X = np.array(
            [
                [-3.0, -0.1],
                [3.0, 0.1],
                [-3.0, 0.1],
                [3.0, -0.1],
            ]
        )
        model = PCA(n_components=2).fit(X)
        # first component along x, second along y (up to sign)
        np.testing.assert_allclose(np.abs(model.components_[0]), [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(np.abs(model.components_[1]), [0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(
            model.explained_variance_, [X[:, 0].var(ddof=1), X[:, 1].var(ddof=1)]
        )


class TestOrthonormality:
    def test_components_are_orthonormal(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(50, 6))
        model = PCA(n_components=4).fit(X)
        gram = model.components_ @ model.components_.T
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-10)


class TestObjectiveEquivalence:
    def test_projected_variance_plus_reconstruction_error_is_total_variance(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 5))
        k = 3
        model = PCA(n_components=k).fit(X)
        full = PCA(n_components=5).fit(X)

        total_var = np.sum((X - X.mean(axis=0)) ** 2) / (X.shape[0] - 1)
        captured_var = model.explained_variance_.sum()
        reconstructed = model.inverse_transform(model.transform(X))
        residual_var = np.sum((X - reconstructed) ** 2) / (X.shape[0] - 1)

        np.testing.assert_allclose(captured_var + residual_var, total_var, atol=1e-8)
        # keeping every component reconstructs X exactly (residual == 0)
        full_reconstructed = full.inverse_transform(full.transform(X))
        np.testing.assert_allclose(full_reconstructed, X, atol=1e-8)


class TestPowerIterationEquivalence:
    def test_power_iteration_matches_svd_components_and_eigenvalues(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(150, 5)) @ rng.normal(size=(5, 5))
        model = PCA(n_components=4).fit(X)
        components, eigenvalues = _power_iteration_pca(X, 4, np.random.default_rng(123))
        aligned = _align_signs(components, model.components_)
        np.testing.assert_allclose(aligned, model.components_, atol=1e-6)
        np.testing.assert_allclose(eigenvalues, model.explained_variance_, atol=1e-6)

    def test_fix_signs_matches_manual_flip(self) -> None:
        U = np.eye(2)
        Vt = np.array([[-1.0, 0.0], [0.0, -1.0]])
        _fix_signs(U, Vt)
        np.testing.assert_allclose(Vt, np.eye(2))
        np.testing.assert_allclose(U, -np.eye(2))


class TestExplainedVarianceRatio:
    def test_sums_to_one_when_keeping_all_components(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(40, 4))
        model = PCA(n_components=4).fit(X)
        assert model.explained_variance_ratio_.sum() == pytest.approx(1.0)

    def test_non_increasing_and_in_unit_interval(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 5))
        model = PCA(n_components=5).fit(X)
        ratios = model.explained_variance_ratio_
        assert all(a >= b - 1e-12 for a, b in zip(ratios, ratios[1:], strict=False))
        assert np.all(ratios >= 0.0) and np.all(ratios <= 1.0)


class TestDeterminism:
    def test_same_data_same_result(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(30, 4))
        a = PCA(n_components=3).fit(X)
        b = PCA(n_components=3).fit(X)
        np.testing.assert_array_equal(a.components_, b.components_)
        np.testing.assert_array_equal(a.explained_variance_, b.explained_variance_)


class TestContract:
    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            PCA().transform(np.zeros((2, 2)))

    def test_inverse_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            PCA().inverse_transform(np.zeros((2, 1)))

    def test_fit_returns_self(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(10, 3))
        model = PCA(n_components=2)
        assert model.fit(X) is model

    def test_fit_does_not_mutate_hyperparameters(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(10, 3))
        model = PCA(n_components=2)
        model.fit(X)
        assert model.get_params() == {"n_components": 2}

    def test_transform_rejects_wrong_feature_count(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(20, 3))
        model = PCA(n_components=2).fit(X)
        with pytest.raises(ValueError, match="features"):
            model.transform(np.zeros((4, 5)))

    def test_inverse_transform_rejects_wrong_component_count(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(20, 3))
        model = PCA(n_components=2).fit(X)
        with pytest.raises(ValueError, match="n_components_"):
            model.inverse_transform(np.zeros((4, 3)))

    def test_n_components_too_small_raises(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="n_components must be between"):
            PCA(n_components=0).fit(X)

    def test_n_components_too_large_raises(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(10, 3))
        with pytest.raises(ValueError, match="n_components must be between"):
            PCA(n_components=4).fit(X)

    def test_none_keeps_min_of_samples_and_features(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(5, 8))
        model = PCA().fit(X)
        assert model.n_components_ == 5

    def test_fit_transform_matches_fit_then_transform(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(20, 4))
        a = PCA(n_components=2).fit(X).transform(X)
        b = PCA(n_components=2).fit_transform(X)
        np.testing.assert_allclose(a, b)

    def test_score_is_negative_reconstruction_error(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.normal(size=(20, 4)) @ rng.normal(size=(4, 4))
        model = PCA(n_components=4).fit(X)
        assert model.score(X) == pytest.approx(0.0, abs=1e-8)
        partial = PCA(n_components=2).fit(X)
        assert partial.score(X) < 0.0

    def test_repr_round_trips(self) -> None:
        assert repr(PCA(n_components=3)) == "PCA(n_components=3)"
        assert repr(PCA()) == "PCA(n_components=None)"


class TestEdgeCases:
    def test_n_components_one(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(20, 5))
        model = PCA(n_components=1).fit(X)
        assert model.components_.shape == (1, 5)

    def test_single_feature(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(20, 1))
        model = PCA(n_components=1).fit(X)
        np.testing.assert_allclose(np.abs(model.components_), [[1.0]])
        np.testing.assert_allclose(model.explained_variance_[0], X.var(ddof=1))

    def test_already_centered_data(self, rng: np.random.Generator) -> None:
        X = rng.normal(size=(30, 3))
        X = X - X.mean(axis=0)
        model = PCA(n_components=2).fit(X)
        np.testing.assert_allclose(model.mean_, np.zeros(3), atol=1e-10)

    def test_constant_feature_gives_near_zero_variance_no_nan(
        self, rng: np.random.Generator
    ) -> None:
        X = np.column_stack([rng.normal(size=30), np.full(30, 5.0)])
        model = PCA(n_components=2).fit(X)
        assert not np.any(np.isnan(model.components_))
        assert not np.any(np.isnan(model.explained_variance_))
        # the constant feature contributes ~0 variance, captured by the
        # smaller of the two eigenvalues
        assert min(model.explained_variance_) == pytest.approx(0.0, abs=1e-10)
