"""scikit-learn parity for DecisionTreeRegressor.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Targets are smooth, **noise-free** functions of continuous features, so on
any node holding a reasonable number of rows the best split is unique
(equal-gain ties are measure-zero) and predictions match scikit-learn
exactly.

**Exact parity has a floor, and it is the node size, not the depth.** Once
the recursion reaches a handful of rows, ties stop being measure-zero: a
4-row node whose best split isolates one sample can usually be cut on
*several* features to produce that same partition, and the variance
decrease depends only on the partition, so those features tie to the last
bit. We break such a tie towards the lowest feature index; scikit-learn
draws features in a random permutation (it does this even when
``max_features`` is ``None``) and keeps the first, so the two disagree.
On the fixtures below that floor is crossed at depth 6. A second, smaller
divergence rides along: scikit-learn casts ``X`` to ``float32`` internally,
so its thresholds are float32 midpoints while ours are float64 — a ~1e-8
band around every boundary in which a query row can be routed differently.

So the exact-prediction tests here stay shallow (or gate node size with
``min_samples_leaf``), and the fully-grown and ``max_features`` cases
assert the tolerance-level agreement that is actually available. This
mirrors the same tie-break gotcha documented for the classifier in
``docs/derivations/decision_tree.md`` §3a.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.metrics import r2_score
from scratchgrad.tree import DecisionTreeRegressor

pytestmark = pytest.mark.reference


def _data(n, d, seed):
    """A smooth, noise-free target in which every feature contributes."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, d))
    coef = np.linspace(1.0, 0.4, d)  # decreasing, but no feature is pure noise
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1 % d] ** 2 + 0.2 * (X * coef).sum(axis=1)
    return X, y


def _n_leaves(node):
    """Count the leaves under ``node``."""
    if node.left is None:
        return 1
    return _n_leaves(node.left) + _n_leaves(node.right)


@pytest.mark.parametrize("max_depth", [1, 3, 5])
def test_predict_matches_sklearn(max_depth) -> None:
    # Depth 5 is the deepest these fixtures stay tie-free; see the module
    # docstring and test_fully_grown_tracks_sklearn for what happens below.
    tree = pytest.importorskip("sklearn.tree")

    X, y = _data(240, 4, 0)
    X_query, _ = _data(60, 4, 1)

    ours = DecisionTreeRegressor(max_depth=max_depth).fit(X, y)
    theirs = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, y)

    np.testing.assert_allclose(
        ours.predict(X_query), theirs.predict(X_query), rtol=1e-9, atol=1e-9
    )


def test_fully_grown_tracks_sklearn() -> None:
    # Grown to purity the two trees agree on everything that is well defined
    # — same shape, same exact interpolation of the training set, held-out R^2
    # within noise — but not on which of several tied features cut a 4-row
    # node, so the individual held-out predictions are not comparable.
    tree = pytest.importorskip("sklearn.tree")

    X, y = _data(240, 4, 0)
    X_query, y_query = _data(60, 4, 1)

    ours = DecisionTreeRegressor().fit(X, y)
    theirs = tree.DecisionTreeRegressor(random_state=0).fit(X, y)

    # every row gets its own leaf, so both reproduce the training targets
    np.testing.assert_allclose(ours.predict(X), y, rtol=0, atol=1e-12)
    assert ours.max_depth_ == theirs.get_depth()
    assert _n_leaves(ours.tree_) == theirs.get_n_leaves() == X.shape[0]

    ours_r2 = r2_score(y_query, ours.predict(X_query))
    assert abs(ours_r2 - theirs.score(X_query, y_query)) < 0.05


def test_feature_importances_track_sklearn() -> None:
    tree = pytest.importorskip("sklearn.tree")

    # Predictions match, but feature_importances_ can still diverge: when two
    # features give an equal-gain split, our deterministic "lowest index"
    # rule and scikit-learn's RNG feature permutation can attribute that
    # node's variance decrease to different features. Close, not equal.
    X, y = _data(300, 5, 2)
    ours = DecisionTreeRegressor(max_depth=6).fit(X, y)
    theirs = tree.DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y)

    assert ours.feature_importances_.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        ours.feature_importances_, theirs.feature_importances_, atol=0.05
    )


def test_min_samples_leaf_matches_sklearn() -> None:
    tree = pytest.importorskip("sklearn.tree")

    X, y = _data(200, 3, 3)
    X_query, _ = _data(50, 3, 4)

    ours = DecisionTreeRegressor(min_samples_leaf=8).fit(X, y)
    theirs = tree.DecisionTreeRegressor(min_samples_leaf=8, random_state=0).fit(X, y)

    np.testing.assert_allclose(
        ours.predict(X_query), theirs.predict(X_query), rtol=1e-9, atol=1e-9
    )


def test_shallow_sample_weight_matches_sklearn() -> None:
    # A shallow weighted tree: the best split at each node is still unique,
    # so predictions match scikit-learn exactly.
    tree = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(0)

    X, y = _data(240, 4, 0)
    X_query, _ = _data(60, 4, 1)
    w = rng.uniform(0.2, 3.0, size=X.shape[0])

    ours = DecisionTreeRegressor(max_depth=2).fit(X, y, sample_weight=w)
    theirs = tree.DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        X, y, sample_weight=w
    )

    np.testing.assert_allclose(
        ours.predict(X_query), theirs.predict(X_query), rtol=1e-9, atol=1e-9
    )


def test_deep_sample_weight_tracks_sklearn_r2() -> None:
    # Deeper down, a reweighted node often has several equal-gain splits, so
    # our "lowest index" tie-break and scikit-learn's RNG permutation pick
    # different splits — only held-out R^2 is comparable, as with max_features.
    tree = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(1)

    X, y = _data(400, 5, 5)
    X_test, y_test = _data(200, 5, 6)
    w = rng.uniform(0.2, 3.0, size=X.shape[0])

    ours = DecisionTreeRegressor(max_depth=6).fit(X, y, sample_weight=w)
    theirs = tree.DecisionTreeRegressor(max_depth=6, random_state=0).fit(
        X, y, sample_weight=w
    )

    assert ours.feature_importances_.sum() == pytest.approx(1.0)
    ours_r2 = r2_score(y_test, ours.predict(X_test))
    assert abs(ours_r2 - theirs.score(X_test, y_test)) < 0.1


def test_max_features_tracks_sklearn_r2() -> None:
    # With max_features set, our NumPy Generator and scikit-learn's internal
    # C RNG draw different feature subsets, so predictions cannot match
    # exactly. Nor can a *single* pair of trees be compared: one subsampled
    # depth-6 tree on 3-of-10 features has an R^2 standard deviation of ~0.18
    # across seeds, which swamps any real difference (on some seeds we beat
    # scikit-learn by 0.26, on others we trail by 0.35 — noise, not bias).
    # Averaging over seeds is the comparison that means something: the two
    # estimators should have the same expected skill.
    tree = pytest.importorskip("sklearn.tree")

    X, y = _data(400, 10, 5)
    X_test, y_test = _data(200, 10, 6)
    seeds = range(10)

    ours_r2 = np.mean(
        [
            r2_score(
                y_test,
                DecisionTreeRegressor(max_depth=6, max_features="sqrt", random_state=s)
                .fit(X, y)
                .predict(X_test),
            )
            for s in seeds
        ]
    )
    theirs_r2 = np.mean(
        [
            tree.DecisionTreeRegressor(max_depth=6, max_features="sqrt", random_state=s)
            .fit(X, y)
            .score(X_test, y_test)
            for s in seeds
        ]
    )

    assert abs(ours_r2 - theirs_r2) < 0.1
