# Random forest (classification)

The finalized walkthrough for `scratchgrad.ensemble.RandomForestClassifier`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and builds directly on
[`decision_tree.md`](decision_tree.md).

A random forest has no objective function and no gradient — it is an
*ensemble* of decision trees, each fit by the greedy recursion of
[`decision_tree.md`](decision_tree.md). Like the KNN and decision-tree
walkthroughs, this document stands in for a derivation: the variance
argument that motivates bagging, the two randomisation mechanisms, the
aggregation rule, and the out-of-bag estimate.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$, $y$ | $(n, d)$, $(n,)$ | training design matrix, class labels |
| $\mathcal{C}$ | $(K,)$ | `classes_` $= \mathrm{np.unique}(y)$, sorted |
| $B$ | scalar | number of trees (`n_estimators`) |
| $\mathcal{D}^{*b}$ | — | the $b$-th bootstrap resample of the $n$ training rows |
| $T_b$ | — | the decision tree fit on $\mathcal{D}^{*b}$ |
| $m$ | scalar | features considered per split (`max_features`), $m \le d$ |
| $\hat{p}_b(x)$ | $(K,)$ | class-probability vector tree $b$ predicts for $x$ (its leaf's frequency vector) |
| $\mathrm{oob}(i)$ | — | $\{\,b : i \notin \mathcal{D}^{*b}\,\}$ — the trees for which row $i$ is *out-of-bag* |

Naturally multiclass (`classes_` has no size limit). Numeric features only,
same as the base tree.

## 2. Why bagging — the variance argument

A fully grown CART tree is **low bias, high variance**: it drives training
impurity to $0$ and small changes in the training set produce very
different trees ([`decision_tree.md`](decision_tree.md) §5). Averaging many
such trees keeps the low bias and cuts the variance.

Take $B$ identically distributed estimators, each with variance $\sigma^2$
and pairwise correlation $\rho$. The variance of their mean is

$$\mathrm{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right)
  = \rho\,\sigma^2 \;+\; \frac{1 - \rho}{B}\,\sigma^2 .$$

As $B \to \infty$ the second term vanishes; the first term $\rho\sigma^2$
is a floor set by **how correlated the trees are**. So the whole game is
driving $\rho$ down without inflating $\sigma^2$ or the bias. Two
mechanisms do it:

1. **Bootstrap resampling** (Breiman, 1996 — *bagging*): each tree is fit
   on its own resample $\mathcal{D}^{*b}$ of $n$ rows drawn *with
   replacement* (§3).
2. **Per-split feature subsampling** (Breiman, 2001 — the "random" in
   random forest): at every node, only a fresh random subset of $m$
   features is a split candidate (§4). This is what stops a few dominant
   predictors from forcing the same splits near the root of every tree —
   the single biggest lever on $\rho$.

Bias is essentially unchanged: each tree is still (near) unbiased for the
Bayes-optimal partition, so their average is too. The forest trades a
negligible bias increase for a large variance reduction — a favourable
move whenever the base learner is an overfit tree.

## 3. Bootstrap and out-of-bag samples

$\mathcal{D}^{*b}$: draw $n$ row indices uniformly **with replacement**
from $\{1, \dots, n\}$. The probability that a given row $i$ is never drawn
in one resample is

$$\left(1 - \frac{1}{n}\right)^{n} \xrightarrow[n \to \infty]{} e^{-1}
  \approx 0.368 .$$

So on average ~37% of the rows are **out-of-bag (OOB)** for each tree —
they took no part in fitting it. Collecting, for each training row $i$,
the predictions of just the trees in $\mathrm{oob}(i)$ gives a
validation estimate for free:

$$\hat{p}^{\text{oob}}(x_i) =
  \frac{1}{\lvert\mathrm{oob}(i)\rvert}
  \sum_{b \in \mathrm{oob}(i)} \hat{p}_b(x_i),
  \qquad
  \hat{y}^{\text{oob}}(x_i) = \mathcal{C}\big[\arg\max_k \hat{p}^{\text{oob}}(x_i)_k\big]$$

and `oob_score_` is the accuracy of $\hat{y}^{\text{oob}}$ over the rows
that were OOB for at least one tree. For large $B$ this tracks
leave-one-out cross-validation closely, at no extra fitting cost. It is
only defined when `bootstrap=True` (with `bootstrap=False` every row is in
every tree — `oob_score=True` then raises).

A row that is *in-bag for every tree* (possible, probability
$\approx (1 - e^{-1})^B$, tiny for the default $B = 100$) has no OOB
prediction; those rows are excluded from `oob_score_` and their row of
`oob_decision_function_` is `nan`.

## 4. Feature subsampling

Handled entirely by the base tree — see
[`decision_tree.md`](decision_tree.md) §3a. The forest passes
`max_features` straight through to every `DecisionTreeClassifier` it
builds; the default is `"sqrt"` ($m = \lfloor\sqrt{d}\rfloor$), Breiman's
recommendation for classification. Each node of each tree redraws its own
size-$m$ candidate subset from the tree's seeded `Generator`.

Because our `Generator` stream and scikit-learn's internal C RNG differ,
a seeded forest is reproducible **against itself** but its individual
predictions cannot be expected to match scikit-learn's tree-for-tree —
only its held-out accuracy, to a tolerance.

## 5. Aggregation — soft voting

Each tree outputs a probability vector $\hat{p}_b(x)$ (the class
frequencies of the leaf $x$ lands in). The forest **averages the
probability vectors** and takes the argmax — scikit-learn's rule, and the
reason parity is even approachable:

$$\hat{p}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(x),
  \qquad
  \hat{y}(x) = \mathcal{C}\big[\arg\max_k \hat{p}(x)_k\big].$$

This "soft" vote uses each tree's confidence, unlike a "hard" majority
vote over the $\arg\max$ labels (which scikit-learn does *not* use).
Argmax ties resolve to the lowest class label (`classes_` is sorted), as
everywhere else in the codebase.

**A bootstrap can drop a rare class.** If class $k$ never appears in
$\mathcal{D}^{*b}$, tree $b$'s `classes_` is missing it and its
`predict_proba` has $K - 1$ columns. The forest fixes this at aggregation
time: it scatters each tree's columns into the full $K$-wide space via
`np.searchsorted(forest.classes_, tree.classes_)` (both sorted, tree's
classes $\subseteq$ forest's), leaving $0$ in the absent column, then
averages. No change to the base tree is needed.

## 6. What it costs

- **Fitting** is $O\!\big(B \cdot m\, n \log n \cdot \text{depth}\big)$ —
  $B$ independent tree builds. Done **serially**: there is no `n_jobs`.
  Parallelism is an engineering concern, not part of the method, and
  would clutter the lesson (same spirit as KNN having no KD-tree and
  Lasso no `solver`).
- **Memory** is $B$ fitted trees.
- **Interpretability** drops relative to one tree; `feature_importances_`
  (the mean of the per-tree Gini importances, renormalised to sum to 1)
  is the usual summary and is more stable than a single tree's.
- **It does not overfit as $B$ grows.** Adding trees drives the variance
  term $\frac{1-\rho}{B}\sigma^2 \to 0$ and then stops helping; it never
  increases test error. $B$ is a compute/accuracy tradeoff, not a
  regulariser — unlike `max_depth`. The tree-level regularisers still
  apply per tree but forests are usually run with deep (unbounded) trees.

## 7. Algorithm (pseudocode)

```
RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       min_impurity_decrease=0.0, max_features="sqrt",
                       bootstrap=True, oob_score=False, random_state=None)

fit(X, y):
    validate: n_estimators >= 1;  not (oob_score and not bootstrap)
    X, y = check_X_y(X, y)
    classes_ = unique(y);  y_idx = searchsorted(classes_, y)
    n = len(y)
    seeds = SeedSequence(random_state).spawn(n_estimators)   # independent per tree
    estimators_, oob_masks = [], []
    for seed in seeds:
        rng_b = default_rng(seed)
        if bootstrap:
            rows = rng_b.integers(0, n, size=n)              # n draws, with replacement
            in_bag = zeros(n, bool);  in_bag[rows] = True
        else:
            rows = arange(n);  in_bag = ones(n, bool)
        tree = DecisionTreeClassifier(criterion, max_depth, min_samples_split,
                                      min_samples_leaf, min_impurity_decrease,
                                      max_features=max_features,
                                      random_state=rng_b)     # same stream, feature draws
        tree.fit(X[rows], y[rows])
        estimators_.append(tree);  oob_masks.append(~in_bag)
    feature_importances_ = normalise(mean(t.feature_importances_ for t in estimators_))
    if oob_score:
        oob_decision_function_, oob_score_ = _oob_score(estimators_, oob_masks,
                                                        X, y, classes_)
    return self

_aggregate_proba(estimators_, classes_, X):                  # -> (m, K)
    total = zeros((len(X), len(classes_)))
    for tree in estimators_:
        cols = searchsorted(classes_, tree.classes_)         # full-space columns
        total[:, cols] += tree.predict_proba(X)
    return total / len(estimators_)

_oob_score(estimators_, oob_masks, X, y, classes_):
    agg = zeros((n, K));  count = zeros(n)
    for tree, oob in zip(estimators_, oob_masks):
        if not oob.any():  continue
        full = zeros((oob.sum(), K))
        full[:, searchsorted(classes_, tree.classes_)] = tree.predict_proba(X[oob])
        agg[oob] += full                                      # boolean-mask add
        count[oob] += 1
    scored = count > 0
    decision = agg / count[:, None]                           # nan where count == 0
    pred = classes_[argmax(decision[scored], axis=1)]
    return decision, accuracy_score(y[scored], pred)

predict_proba(X):  check_is_fitted; check_array; feature-count check
                   return _aggregate_proba(estimators_, classes_, X)
predict(X):        classes_[argmax(predict_proba(X), axis=1)]
score(X, y):       accuracy_score(y, predict(X))
```

Module-level `_aggregate_proba` (and `_oob_score`) are factored out so the
vote and the OOB estimate are unit-tested directly, mirroring the `_*`
helpers in `linear/`, `neighbors/`, `naive_bayes/`, and `tree/`.

## 8. What the tests check

| Tier | Check |
| --- | --- |
| Variance reduction | On a noisy synthetic set: forest held-out accuracy $\ge$ a single unbounded tree's; the spread of forest predictions across seeds is smaller than the spread of single-tree predictions across seeds. |
| Bootstrap | $(1 - 1/n)^n \to 1/e$ numerically; each bag has exactly $n$ rows; OOB fraction $\approx 0.37$; every row's `oob` mask is `~in_bag`. |
| OOB estimate | `oob_score_` is within a small tolerance of a genuine held-out accuracy on the same model; `oob_score=True` with `bootstrap=False` raises; `oob_decision_function_` rows sum to 1 (or are `nan`), shape $(n, K)$. |
| Aggregation | `predict_proba` equals an independent mean of the per-tree (column-aligned) probabilities; rows sum to 1; shape $(m, K)$; `predict == classes_[argmax(proba)]`. `_aggregate_proba` handles a tree whose `classes_` is missing a label (constructed bootstrap that drops a class). |
| Determinism | Same `random_state` → identical `estimators_` (structural) and identical predictions; different seed → generally different forest. |
| Reduces to a tree | `n_estimators=1, bootstrap=False, max_features=None, random_state=k` gives predictions identical to a lone `DecisionTreeClassifier(random_state=k)` on the same data. |
| `max_features` passthrough | each fitted tree's `max_features_` matches what the forest was given; default resolves to `"sqrt"`. |
| Contract | `NotFittedError` before `fit`; `fit` returns self; hyperparameters unmutated; feature-count mismatch in `predict`/`predict_proba` raises; `n_estimators < 1` raises; unknown `max_features` string raises (via the tree); `repr` round-trips. |
| Behavioral | `make_moons(noise=0.3)` and 3-class `make_blobs`: high held-out accuracy, beats a `LogisticRegression` baseline; accuracy is non-decreasing (within noise) as `n_estimators` grows from 1 → 50. |
| Reference (`-m reference`) | held-out accuracy within a tolerance of `sklearn.ensemble.RandomForestClassifier` with matched hyperparameters (exact predictions are not expected — the RNG streams differ); `oob_score_` within `atol` of scikit-learn's. |

Plus `examples/random_forest.py` — seeded, runnable: single-tree vs forest
test accuracy as $B$ grows, the OOB-score curve, and the averaged
`feature_importances_` against a single tree's.

## Scope — what this estimator omits (and why)

- **`RandomForestRegressor`** — a follow-up one-file PR once
  `DecisionTreeRegressor` exists (MSE criterion + mean-value leaves +
  averaging instead of voting), exactly as `KNeighborsRegressor` was
  deferred after the classifier.
- **`n_jobs` / parallelism**, **`warm_start`**, **`max_samples`** (bag
  smaller than $n$), **`class_weight`**, **`ccp_alpha`**,
  **`min_weight_fraction_leaf`** — standard scikit-learn knobs left out to
  keep the lesson (decorrelated bagging + OOB) uncluttered.
- **ExtraTrees** (random split thresholds, no bootstrap by default) — a
  different ensemble, out of scope.

## References

- Breiman (2001), "Random Forests", *Machine Learning* 45(1).
- Breiman (1996), "Bagging Predictors", *Machine Learning* 24(2).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §15 (Random Forests) and §8.7 (Bagging).
- scikit-learn User Guide, §1.11.2 (Forests of randomized trees).
