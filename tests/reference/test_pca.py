"""scikit-learn parity for PCA.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

PCA has no RNG anywhere in its `fit` (a deterministic SVD, with the sign
ambiguity resolved the same way scikit-learn resolves it -- see
docs/derivations/pca.md section 5), so unlike the k-means++/EM-with-random-init
paths elsewhere in this project, this is **exact** parity, not a tolerance
comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.decomposition import PCA

pytestmark = pytest.mark.reference


def test_matches_sklearn_exactly() -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6)) @ rng.normal(size=(6, 6))

    ours = PCA(n_components=4).fit(X)
    theirs = decomposition.PCA(n_components=4, svd_solver="full").fit(X)

    np.testing.assert_allclose(ours.components_, theirs.components_, atol=1e-8)
    np.testing.assert_allclose(
        ours.explained_variance_, theirs.explained_variance_, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.singular_values_, theirs.singular_values_, atol=1e-8
    )
    np.testing.assert_allclose(ours.transform(X), theirs.transform(X), atol=1e-6)


def test_matches_sklearn_with_all_components_kept() -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 5))

    ours = PCA().fit(X)
    theirs = decomposition.PCA(svd_solver="full").fit(X)

    np.testing.assert_allclose(ours.components_, theirs.components_, atol=1e-8)
    np.testing.assert_allclose(ours.explained_variance_ratio_.sum(), 1.0, atol=1e-8)
    inverse_ours = ours.inverse_transform(ours.transform(X))
    np.testing.assert_allclose(inverse_ours, X, atol=1e-8)


def test_inverse_transform_matches_sklearn() -> None:
    decomposition = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4))

    ours = PCA(n_components=2).fit(X)
    theirs = decomposition.PCA(n_components=2, svd_solver="full").fit(X)

    ours_recon = ours.inverse_transform(ours.transform(X))
    theirs_recon = theirs.inverse_transform(theirs.transform(X))
    np.testing.assert_allclose(ours_recon, theirs_recon, atol=1e-6)
