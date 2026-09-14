"""scikit-learn parity for GaussianMixture.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Unlike KMeans's explicit ``init`` ndarray, this project's ``GaussianMixture``
exposes no fully deterministic, RNG-free ``init_params`` path (see
``docs/derivations/gaussian_mixture.md`` §11 scope note). scikit-learn's
constructor *does* accept explicit ``weights_init``/``means_init``/
``precisions_init``, so the exact-parity strategy here is: seed both this
project's module-level EM step functions and
``sklearn.mixture.GaussianMixture`` from an *identical* fixed starting
point, then run a fixed number of EM iterations with ``tol=0`` (so neither
side can stop early) and compare. This isolates the E/M formulas themselves
from initialization randomness — the GaussianMixture analogue of KMeans's
"explicit init array" exact-parity test.

With the default ``init_params="kmeans"`` path (real RNG divergence from
sklearn, both in the seeding KMeans run and in EM's own initialization),
parity is tolerance-only: comparable average log-likelihood across seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.cluster import GaussianMixture
from scratchgrad.cluster.gaussian_mixture import _em, _m_step
from scratchgrad.datasets import make_blobs

pytestmark = pytest.mark.reference


def _precisions_init(covariances: np.ndarray, covariance_type: str) -> np.ndarray:
    """Convert a fixed starting covariance into sklearn's ``precisions_init``."""
    if covariance_type in ("full", "tied"):
        return np.linalg.inv(covariances)
    return 1.0 / covariances  # diag / spherical: elementwise reciprocal


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_fixed_start_em_matches_sklearn_exactly(covariance_type: str) -> None:
    mixture = pytest.importorskip("sklearn.mixture")

    X, _ = make_blobs(
        n_samples=200, n_features=3, centers=3, cluster_std=1.5, random_state=1
    )
    n_components = 3
    rng = np.random.default_rng(0)

    resp0 = rng.uniform(size=(X.shape[0], n_components))
    resp0 /= resp0.sum(axis=1, keepdims=True)
    weights0, means0, covariances0 = _m_step(X, resp0, covariance_type, reg_covar=1e-6)

    weights, means, covariances, lower_bound, n_iter, _converged = _em(
        X,
        weights0,
        means0,
        covariances0,
        covariance_type,
        max_iter=5,
        tol=0.0,
        reg_covar=1e-6,
    )

    theirs = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        weights_init=weights0,
        means_init=means0,
        precisions_init=_precisions_init(covariances0, covariance_type),
        max_iter=5,
        tol=0.0,
        n_init=1,
        reg_covar=1e-6,
    ).fit(X)

    assert n_iter == theirs.n_iter_
    np.testing.assert_allclose(weights, theirs.weights_, atol=1e-8)
    np.testing.assert_allclose(means, theirs.means_, atol=1e-6)
    np.testing.assert_allclose(covariances, theirs.covariances_, atol=1e-5)
    assert lower_bound == pytest.approx(theirs.lower_bound_, abs=1e-8)


def test_default_kmeans_init_tracks_sklearn_log_likelihood() -> None:
    mixture = pytest.importorskip("sklearn.mixture")

    X, _ = make_blobs(
        n_samples=300, n_features=4, centers=4, cluster_std=2.0, random_state=2
    )

    ours = GaussianMixture(n_components=4, n_init=5, random_state=0).fit(X)
    theirs = mixture.GaussianMixture(n_components=4, n_init=5, random_state=0).fit(X)

    assert (
        abs(ours.lower_bound_ - theirs.lower_bound_) / abs(theirs.lower_bound_) < 0.05
    )


def test_predict_on_held_out_data_tracks_sklearn_log_likelihood() -> None:
    mixture = pytest.importorskip("sklearn.mixture")

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=0)
    X_test, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=1)

    ours = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    theirs = mixture.GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)

    ours_score = ours.score(X_test)
    theirs_score = theirs.score(X_test)
    assert abs(ours_score - theirs_score) / abs(theirs_score) < 0.05


def test_bic_tracks_sklearn() -> None:
    mixture = pytest.importorskip("sklearn.mixture")

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=0)

    ours = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
    theirs = mixture.GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)

    assert abs(ours.bic(X) - theirs.bic(X)) / abs(theirs.bic(X)) < 0.05
