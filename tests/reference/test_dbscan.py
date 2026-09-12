"""scikit-learn parity for DBSCAN.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Unlike every prior reference test module here, DBSCAN has **no RNG
anywhere** in its own definition -- no ``init``, no bootstrap, no feature
subsampling -- so parity is exact on every input, not just a deterministic
special case. ``sklearn.cluster.DBSCAN`` is constructed with
``algorithm="brute"`` so both sides visit each point's neighbor list in the
same ascending-index order, which is what the traversal in
``docs/derivations/dbscan.md`` section 4 depends on for the border-point
tie to resolve identically on both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.cluster import DBSCAN
from scratchgrad.datasets import make_blobs, make_moons

pytestmark = pytest.mark.reference


def test_matches_sklearn_exactly_across_several_seeds() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    for seed in range(10):
        X, _ = make_blobs(
            n_samples=150, n_features=2, centers=4, cluster_std=1.5, random_state=seed
        )
        ours = DBSCAN(eps=0.8, min_samples=5).fit(X)
        theirs = cluster.DBSCAN(eps=0.8, min_samples=5, algorithm="brute").fit(X)

        np.testing.assert_array_equal(ours.labels_, theirs.labels_)
        np.testing.assert_array_equal(
            ours.core_sample_indices_, theirs.core_sample_indices_
        )
        np.testing.assert_allclose(ours.components_, theirs.components_)


def test_matches_sklearn_exactly_on_non_convex_moons() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
    for eps in (0.1, 0.15, 0.2, 0.3):
        ours = DBSCAN(eps=eps, min_samples=5).fit(X)
        theirs = cluster.DBSCAN(eps=eps, min_samples=5, algorithm="brute").fit(X)
        np.testing.assert_array_equal(ours.labels_, theirs.labels_)
        np.testing.assert_array_equal(
            ours.core_sample_indices_, theirs.core_sample_indices_
        )


def test_matches_sklearn_exactly_with_manhattan_metric() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    for seed in range(5):
        X, _ = make_blobs(
            n_samples=150, n_features=2, centers=4, cluster_std=1.5, random_state=seed
        )
        ours = DBSCAN(eps=2.0, min_samples=5, metric="manhattan").fit(X)
        theirs = cluster.DBSCAN(
            eps=2.0, min_samples=5, metric="manhattan", algorithm="brute"
        ).fit(X)
        np.testing.assert_array_equal(ours.labels_, theirs.labels_)
        np.testing.assert_array_equal(
            ours.core_sample_indices_, theirs.core_sample_indices_
        )


def test_matches_sklearn_exactly_on_a_deliberate_double_border_case() -> None:
    """Exercises section 4's tie: one point equidistant-ish from two clusters' cores.

    Two 4-point core clusters (a line of points spreading away from a
    "hub" nearest the gap) placed far enough apart that they never connect
    directly, plus one point sitting exactly `eps`-close to *each* hub --
    within `min_samples - 1` of its own neighbors, so it's a border point,
    not core, and the two clusters stay genuinely distinct rather than
    merging into one. This is the exact scenario where the definition
    itself (section 2) has no unique answer for that one point, and
    section 4's traversal-order match with sklearn is what makes both
    sides land on the identical label for it.
    """
    cluster = pytest.importorskip("sklearn.cluster")

    left_line = np.array([[0.0, 0.0], [-0.2, 0.0], [-0.4, 0.0], [-0.6, 0.0]])
    right_line = np.array([[1.6, 0.0], [1.8, 0.0], [2.0, 0.0], [2.2, 0.0]])
    border_point = np.array([[0.8, 0.0]])  # 0.8 from each line's near hub
    noise_point = np.array([[100.0, 100.0]])
    X = np.vstack([left_line, right_line, border_point, noise_point])

    ours = DBSCAN(eps=0.85, min_samples=4).fit(X)
    theirs = cluster.DBSCAN(eps=0.85, min_samples=4, algorithm="brute").fit(X)

    # sanity: the border point (index 8) really is ambiguous under the
    # definition (section 2) -- neighbor of a core point on both sides --
    # and really is non-core itself, otherwise this test proves nothing.
    assert ours.labels_[8] not in (-1,)
    assert 8 not in ours.core_sample_indices_
    assert len(set(ours.labels_.tolist()) - {-1}) == 2  # noqa: PLR2004

    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    np.testing.assert_array_equal(
        ours.core_sample_indices_, theirs.core_sample_indices_
    )
