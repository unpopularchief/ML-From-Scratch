"""scikit-learn parity for LinearSVM.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

**No exact parity.** ``sklearn.svm.LinearSVC`` solves the *dual* soft-margin
problem via liblinear's coordinate descent; this project's ``LinearSVM``
minimises the same primal objective directly by subgradient descent
(``docs/derivations/linear_svm.md`` §7) -- a different algorithm converging
to a (generally distinct, since the primal isn't strictly convex in every
direction) optimum of the same convex problem. So the comparison is on
outcomes -- held-out accuracy and decision-boundary sign agreement -- within
a tolerance, the same tier ``RandomForest``/``GradientBoosting`` use, not
``coef_``/``intercept_`` equality.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.preprocessing import StandardScaler, train_test_split
from scratchgrad.svm import LinearSVM

pytestmark = pytest.mark.reference


def test_tracks_sklearn_on_separable_blobs() -> None:
    svm = pytest.importorskip("sklearn.svm")

    X, y = make_blobs(n_samples=400, centers=2, cluster_std=1.2, random_state=0)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = LinearSVM(C=1.0, lr=0.1, max_iter=3000).fit(X_tr, y_tr)
    theirs = svm.LinearSVC(C=1.0, loss="hinge", max_iter=10_000, random_state=0).fit(
        X_tr, y_tr
    )

    assert abs(ours.score(X_te, y_te) - theirs.score(X_te, y_te)) < 0.05

    our_sign = np.sign(ours.decision_function(X_te))
    their_sign = np.sign(theirs.decision_function(X_te))
    assert float(np.mean(our_sign == their_sign)) > 0.9


def test_tracks_sklearn_on_noisy_moons() -> None:
    svm = pytest.importorskip("sklearn.svm")

    X, y = make_moons(n_samples=400, noise=0.25, random_state=1)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = LinearSVM(C=1.0, lr=0.1, max_iter=3000).fit(X_tr, y_tr)
    theirs = svm.LinearSVC(C=1.0, loss="hinge", max_iter=10_000, random_state=0).fit(
        X_tr, y_tr
    )

    assert abs(ours.score(X_te, y_te) - theirs.score(X_te, y_te)) < 0.08


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_tracks_sklearn_accuracy_across_C(C) -> None:
    svm = pytest.importorskip("sklearn.svm")

    X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=2)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = LinearSVM(C=C, lr=0.1, max_iter=3000).fit(X_tr, y_tr)
    theirs = svm.LinearSVC(C=C, loss="hinge", max_iter=10_000, random_state=0).fit(
        X_tr, y_tr
    )

    assert abs(ours.score(X_te, y_te) - theirs.score(X_te, y_te)) < 0.08
