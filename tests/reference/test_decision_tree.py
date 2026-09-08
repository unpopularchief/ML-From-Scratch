"""scikit-learn parity for DecisionTreeClassifier.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Data is continuous Gaussian blobs, where the best split at each node is
unique — scikit-learn permutes features with its RNG and so breaks
equal-gain ties randomly, which would make exact structural parity
impossible on tie-prone data.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs
from scratchgrad.tree import DecisionTreeClassifier

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("max_depth", [1, 3, 5, None])
def test_predict_and_proba_match_sklearn(criterion, max_depth) -> None:
    tree = pytest.importorskip("sklearn.tree")

    X, y = make_blobs(
        n_samples=240, n_features=4, centers=3, cluster_std=3.0, random_state=0
    )
    X_query, _ = make_blobs(
        n_samples=60, n_features=4, centers=3, cluster_std=3.0, random_state=1
    )

    ours = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth).fit(X, y)
    theirs = tree.DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=0
    ).fit(X, y)

    np.testing.assert_array_equal(ours.predict(X_query), theirs.predict(X_query))
    np.testing.assert_allclose(
        ours.predict_proba(X_query), theirs.predict_proba(X_query), rtol=1e-6
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_feature_importances_track_sklearn(criterion) -> None:
    tree = pytest.importorskip("sklearn.tree")

    # Both trees make identical predictions here, but feature_importances_ can
    # still diverge: when two features give an equal-gain split, our
    # deterministic "lowest index" rule and scikit-learn's RNG feature
    # permutation can attribute that node's impurity decrease to different
    # features. So this checks the importances are close, not exactly equal.
    X, y = make_blobs(
        n_samples=300, n_features=5, centers=4, cluster_std=2.0, random_state=2
    )
    ours = DecisionTreeClassifier(criterion=criterion, max_depth=6).fit(X, y)
    theirs = tree.DecisionTreeClassifier(
        criterion=criterion, max_depth=6, random_state=0
    ).fit(X, y)

    assert ours.feature_importances_.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        ours.feature_importances_, theirs.feature_importances_, atol=0.05
    )


def test_min_samples_leaf_matches_sklearn() -> None:
    tree = pytest.importorskip("sklearn.tree")

    X, y = make_blobs(
        n_samples=200, n_features=3, centers=2, cluster_std=4.0, random_state=3
    )
    X_query, _ = make_blobs(
        n_samples=50, n_features=3, centers=2, cluster_std=4.0, random_state=4
    )

    ours = DecisionTreeClassifier(min_samples_leaf=8).fit(X, y)
    theirs = tree.DecisionTreeClassifier(min_samples_leaf=8, random_state=0).fit(X, y)

    np.testing.assert_array_equal(ours.predict(X_query), theirs.predict(X_query))


def test_max_features_tracks_sklearn_accuracy() -> None:
    # With max_features set, our NumPy Generator and scikit-learn's internal C
    # RNG produce different feature draws, so predictions cannot match exactly.
    # A subsampled single tree should still land in the same accuracy ballpark.
    tree = pytest.importorskip("sklearn.tree")

    X, y = make_blobs(
        n_samples=400, n_features=10, centers=3, cluster_std=5.0, random_state=5
    )
    X_test, y_test = make_blobs(
        n_samples=200, n_features=10, centers=3, cluster_std=5.0, random_state=6
    )

    ours = DecisionTreeClassifier(max_depth=6, max_features="sqrt", random_state=0)
    ours.fit(X, y)
    theirs = tree.DecisionTreeClassifier(
        max_depth=6, max_features="sqrt", random_state=0
    ).fit(X, y)

    ours_acc = np.mean(ours.predict(X_test) == y_test)
    theirs_acc = theirs.score(X_test, y_test)
    assert abs(ours_acc - theirs_acc) < 0.1
