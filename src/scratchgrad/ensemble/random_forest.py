r"""Random forest — classification.

Bootstrap-aggregated decision trees with per-node feature subsampling
(Breiman, 2001). Each of ``n_estimators`` trees is grown by the CART
recursion of :mod:`scratchgrad.tree` on its own bootstrap resample of the
training rows, and at every split only ``max_features`` randomly chosen
features are candidates. Prediction averages the trees' class-probability
vectors (soft voting) and takes the argmax.

Averaging de-correlated low-bias / high-variance trees keeps the low bias
and drives the variance toward the floor set by the residual tree-to-tree
correlation:

.. math::
    \mathrm{Var}\!\left(\tfrac{1}{B}\textstyle\sum_b T_b\right)
      = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2 .

The out-of-bag rows — the :math:`\approx e^{-1}` fraction left out of each
bootstrap — give a free validation estimate (``oob_score_``). Trees are
built serially (no ``n_jobs``). Full walkthrough:
``docs/derivations/random_forest.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.tree import DecisionTreeClassifier
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)


def _aggregate_proba(
    estimators: list[DecisionTreeClassifier], classes: FloatArray, X: FloatArray
) -> FloatArray:
    r"""Soft vote: mean of the trees' class-probability vectors, shape ``(m, K)``.

    Each tree's ``predict_proba`` columns are scattered into the full
    ``classes`` space before averaging — a bootstrap resample can omit a
    rare class, leaving that tree's ``classes_`` (and proba matrix)
    narrower than the forest's. ``classes`` and every ``tree.classes_`` are
    sorted and ``tree.classes_`` is a subset, so
    :func:`numpy.searchsorted` gives the right target columns.
    """
    total = np.zeros((X.shape[0], classes.shape[0]))
    for tree in estimators:
        cols = np.searchsorted(classes, tree.classes_)
        total[:, cols] += tree.predict_proba(X)
    return total / len(estimators)


def _oob_score(
    estimators: list[DecisionTreeClassifier],
    oob_masks: list[np.ndarray],
    X: FloatArray,
    y: TargetVector,
    classes: FloatArray,
) -> tuple[FloatArray, float]:
    r"""Out-of-bag decision function and accuracy.

    For each training row, average the probability vectors of only the
    trees that did not see it (``oob_masks[b]`` is ``True`` where row is
    out-of-bag for tree ``b``), then argmax. Rows that were in-bag for
    *every* tree get a ``nan`` row in the decision function and are
    excluded from the score.
    """
    n, n_classes = X.shape[0], classes.shape[0]
    agg = np.zeros((n, n_classes))
    count = np.zeros(n)
    for tree, oob in zip(estimators, oob_masks, strict=True):
        if not oob.any():
            continue
        block = np.zeros((int(oob.sum()), n_classes))
        block[:, np.searchsorted(classes, tree.classes_)] = tree.predict_proba(X[oob])
        agg[oob] += block  # boolean-mask add-in-place
        count[oob] += 1

    scored = count > 0
    with np.errstate(invalid="ignore"):
        decision = agg / count[:, None]  # 0/0 -> nan on never-OOB rows
    oob_pred = classes[np.argmax(decision[scored], axis=1)]
    return decision, accuracy_score(y[scored], oob_pred)


class RandomForestClassifier(Estimator):
    r"""A random forest classifier — bagged, feature-subsampled CART trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"gini", "entropy"}, default="gini"
        Split impurity measure, passed to every tree.
    max_depth : int, optional
        Per-tree depth cap. ``None`` grows each tree until its nodes are
        pure or too small to split — the usual forest setting.
    min_samples_split, min_samples_leaf : int, default=2, 1
        Per-tree stopping rules (see
        :class:`scratchgrad.tree.DecisionTreeClassifier`).
    min_impurity_decrease : float, default=0.0
        Per-tree minimum scaled impurity decrease for a split.
    max_features : {"sqrt", "log2"}, int, float, or None, default="sqrt"
        Features considered per split, drawn afresh at every node of every
        tree. ``"sqrt"`` (:math:`\lfloor\sqrt d\rfloor`) is Breiman's
        classification default and the main lever on tree-to-tree
        correlation. ``None`` disables subsampling.
    bootstrap : bool, default=True
        If ``True`` each tree is fit on ``n_samples`` rows drawn with
        replacement; if ``False`` every tree sees all the rows (and
        differs only through feature subsampling).
    oob_score : bool, default=False
        If ``True`` compute ``oob_score_`` / ``oob_decision_function_``
        from the out-of-bag rows. Requires ``bootstrap=True``.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds an independent ``Generator`` per tree (via
        :meth:`numpy.random.Generator.spawn`), driving both the bootstrap
        draw and that tree's per-node feature draws. A fixed value makes
        the whole forest reproducible.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted trees, in build order.
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels. ``predict`` returns values from here;
        ``predict_proba`` columns are in this order.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_importances_ : ndarray of shape (n_features,)
        Mean of the trees' Gini importances, renormalised to sum to 1
        (all zeros if every tree is a single leaf).
    oob_score_ : float
        Out-of-bag accuracy. Only set when ``oob_score=True``.
    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Out-of-bag class probabilities for the training rows; ``nan`` for a
        row that was in-bag for every tree. Only set when
        ``oob_score=True``.

    Notes
    -----
    Trees are built serially — there is no ``n_jobs``. Because the per-node
    feature draws use a NumPy ``Generator`` (not scikit-learn's internal C
    RNG), a seeded forest is reproducible against itself but its individual
    predictions are not expected to match scikit-learn tree-for-tree —
    only its held-out accuracy, to a tolerance. Full walkthrough:
    ``docs/derivations/random_forest.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.ensemble import RandomForestClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    >>> forest.predict(np.array([[3.0], [9.0]])).tolist()
    [0.0, 1.0]

    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X: FeatureMatrix, y: TargetVector) -> RandomForestClassifier:
        """Grow the forest on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_estimators`` is less than 1, or ``oob_score`` is set
            without ``bootstrap``. Per-tree hyperparameters
            (``criterion``, ``max_features``, ...) are validated by the
            first tree and surface the same way.

        """
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True.")

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        tree_rngs = check_random_state(self.random_state).spawn(self.n_estimators)

        self.estimators_ = []
        oob_masks: list[np.ndarray] = []
        for tree_rng in tree_rngs:
            if self.bootstrap:
                rows = tree_rng.integers(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[rows] = True
            else:
                rows = np.arange(n_samples)
                in_bag = np.ones(n_samples, dtype=bool)

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                random_state=tree_rng,  # bootstrap + this tree's feature draws
            )
            tree.fit(X[rows], y[rows])
            self.estimators_.append(tree)
            oob_masks.append(~in_bag)

        importances = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0.0 else importances

        if self.oob_score:
            self.oob_decision_function_, self.oob_score_ = _oob_score(
                self.estimators_, oob_masks, X, y, self.classes_
            )
        return self

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class probabilities, shape ``(n_samples, n_classes)``.

        The mean of the trees' leaf class-frequency vectors (soft voting).
        Rows sum to 1; columns are ordered as ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_}."
            )
        return _aggregate_proba(self.estimators_, self.classes_, X)

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        Returns ``classes_[argmax(predict_proba(X))]``; ties resolve to the
        lowest class label.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        # into a local first — it runs the fitted / feature-count checks, so
        # ``self.classes_`` below can't raise a bare AttributeError pre-fit.
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``."""
        return accuracy_score(y, self.predict(X))
