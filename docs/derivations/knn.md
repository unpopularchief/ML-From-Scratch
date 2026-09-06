# k-Nearest Neighbours (classification)

The finalized walkthrough for `scratchgrad.neighbors.KNeighborsClassifier`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

KNN has no objective function and no training step, so this document takes
the place of a derivation by writing out the decision rule, the statistical
argument for why it works, and the design choices — the same way the
`linear/` derivations do, minus the calculus.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | training design matrix, memorised by `fit` |
| $y$ | $(n,)$ | training labels — any number of classes |
| $X_q$ | $(m, d)$ | query points passed to `predict` |
| $k$ | scalar $\ge 1$ | number of neighbours (`n_neighbors`) |
| $d(\cdot, \cdot)$ | — | distance (`metric`): Euclidean or Manhattan |
| $\mathcal{N}_k(x)$ | — | indices of the $k$ training points closest to $x$ |

`classes_` is `np.unique(y)` (sorted). Predictions are drawn from it. KNN
is naturally multiclass — there is no binary restriction.

## 2. The rule — instance-based, lazy

KNN is **non-parametric** (it assumes no functional form for the decision
boundary) and **lazy** (`fit` does no work beyond storing the data). For a
query point $x$:

1. compute $d(x, x_i)$ for every training point $x_i$;
2. let $\mathcal{N}_k(x)$ be the indices of the $k$ smallest;
3. **predict** the class with the largest vote among
   $\{y_i : i \in \mathcal{N}_k(x)\}$ — each neighbour contributing weight
   $1$ (`weights="uniform"`) or $1/d(x, x_i)$ (`weights="distance"`).

The predicted class-probability vector is the (weighted) fraction of the
$k$ neighbours in each class:

$$\hat{P}(c \mid x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i\,
  \mathbb{1}[y_i = c]}{\sum_{i \in \mathcal{N}_k(x)} w_i},
  \qquad w_i = \begin{cases} 1 & \text{uniform} \\
  1/d(x, x_i) & \text{distance} \end{cases}$$

and `predict` returns $\arg\max_c \hat{P}(c \mid x)$.

## 3. Why it works — Bayes consistency

The Bayes-optimal classifier predicts $\arg\max_c P(c \mid x)$ and attains
the lowest achievable error rate $R^\*$. KNN **estimates** $P(c \mid x)$ by
the local label frequency $\hat{P}(c \mid x)$ above — a piecewise-constant
nonparametric estimate that gets its resolution from how densely the
training data covers the neighbourhood of $x$.

- **Cover & Hart (1967).** As $n \to \infty$ with $k = 1$, the 1-NN error
  rate $R_{1\text{NN}}$ satisfies $R^\* \le R_{1\text{NN}} \le 2R^\*$ — at
  most twice the Bayes error, for *any* distribution.
- **Consistency.** If $k \to \infty$ and $k/n \to 0$ as $n \to \infty$
  (e.g. $k \sim \sqrt{n}$), then $R_{k\text{NN}} \to R^\*$: KNN is
  universally consistent.
- **$k$ is the bias–variance knob.** $k = 1$: zero training error, a jagged
  high-variance boundary that memorises noise. Large $k$: a smoother,
  higher-bias boundary. $k = n$: every query gets the global majority
  class. Odd $k$ avoids two-class ties.

## 4. Two things that bite

**Scaling.** $d(x, x_i)$ mixes all features on their raw numeric scales, so
a feature ranging over $[0, 1000]$ dominates one over $[0, 1]$. KNN is not
scale-invariant. This estimator never rescales its inputs (`plan.md` §1) —
standardise with `scratchgrad.preprocessing.StandardScaler` before `fit`.

**Curse of dimensionality.** In high $d$, pairwise distances concentrate
(nearest and farthest neighbours become nearly equidistant) and "nearby"
loses meaning. KNN is a low-to-moderate-dimension method; no amount of
implementation care fixes this, so it is just documented.

## 5. Computation — brute force

`predict` forms the full $(m, n)$ distance matrix
(`scratchgrad.metrics.pairwise.euclidean_distance` /
`manhattan_distance`, both already vectorised) and selects the $k$ smallest
per row with `np.argpartition(D, k-1, axis=1)[:, :k]` — an $O(mn)$ partial
sort rather than a full $O(mn \log n)$ one. The $k$ selected indices are
then sorted by their distance so tie-breaking and distance weighting are
deterministic.

There is **no `algorithm` parameter**: KD-trees and ball-trees are spatial
acceleration structures, not part of the method's statistics, and would add
a large amount of code that buries the lesson (`plan.md` §6 spirit). This
mirrors `Lasso` having no `solver`. scikit-learn's
`KNeighborsClassifier(algorithm="brute")` is the parity target.

**`weights="distance"` and a zero distance.** If a query coincides with a
training point, $1/d = \infty$. Following scikit-learn: for any query row
that has one or more neighbours at $d = 0$, all weight goes to those
zero-distance neighbours (and the rest of the row's neighbours are
ignored).

**Tie-breaking.** `np.argmax` on the probability row returns the first
maximal entry, so a tie is resolved in favour of the **lowest class label**
(`classes_` is sorted). Deterministic and documented.

## 6. Algorithm (pseudocode)

```
KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")

fit(X, y):
    if metric not in {"euclidean", "manhattan"}: raise ValueError
    if weights not in {"uniform", "distance"}:   raise ValueError
    if n_neighbors < 1:                          raise ValueError
    X, y = check_X_y(X, y)
    if n_neighbors > n_samples:                  raise ValueError   # fail early
    self._X, self._y = X, y
    self.classes_ = unique(y)
    return self

_kneighbors(Xq):                                     # -> dist (m, k), idx (m, k)
    D    = pairwise[metric](Xq, self._X)             # (m, n)
    topk = argpartition(D, n_neighbors - 1, axis=1)[:, :n_neighbors]
    d_topk = take_along_axis(D, topk, axis=1)
    order  = argsort(d_topk, axis=1)                 # sort the k by distance
    idx    = take_along_axis(topk, order, axis=1)
    return take_along_axis(D, idx, axis=1), idx

_vote_proba(labels, weights, classes):              # labels, weights: (m, k)
    proba = zeros((m, len(classes)))
    for j, c in enumerate(classes):
        proba[:, j] = sum(weights * (labels == c), axis=1)
    return proba / proba.sum(axis=1, keepdims=True)

_weights(dist):
    if weights == "uniform": return ones_like(dist)
    with errstate(divide="ignore"): w = 1 / dist
    zero_row = isinf(w).any(axis=1)
    w[zero_row] = isinf(w[zero_row])                 # all weight on the d==0 neighbours
    return w

predict_proba(Xq):
    check_is_fitted(self, "_X")
    Xq = check_array(Xq); validate Xq.shape[1] == self._X.shape[1]
    dist, idx = _kneighbors(Xq)
    return _vote_proba(self._y[idx], _weights(dist), self.classes_)

predict(Xq):  return self.classes_[argmax(predict_proba(Xq), axis=1)]
score(X, y):  return accuracy_score(y, predict(X))
```

Module-level `_vote_proba(labels, weights, classes)` is factored out so the
voting logic is unit-tested directly, mirroring the `_*` helpers in the
`linear/` modules.

## 7. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-laid tiny 2-D set: `_kneighbors` returns the exact indices and distances; a query with a known 3-NN gets the majority label. |
| Independent reference | A brute-force $O(mn)$ pure-Python loop in the test agrees with `_kneighbors` / `predict` on seeded random data (the correctness analogue of a gradient check). |
| $k$ behaviour | `k=1` reproduces training labels on `predict(X_train)` for distinct points; `k = n_samples` predicts the global majority class for every query. |
| weights | Constructed case where the single nearest neighbour is class A but the uniform $k$-vote is class B: `weights="uniform"` → B, `weights="distance"` → A. A query exactly on a training point returns that point's label. |
| Tie-break | Even $k$ with a 50/50 neighbour split predicts the lower class label, deterministically. |
| proba | rows sum to 1; shape `(m, n_classes)`; `predict` equals `classes_[argmax(proba)]`. |
| Multiclass | 3-class `make_blobs`, high held-out accuracy. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; feature-count mismatch in `predict` raises; `n_neighbors > n_samples` raises at `fit`; `n_neighbors < 1` raises; unknown `metric` / `weights` raises; repr round-trips. |
| Behavioral | On noisy `make_moons`, held-out accuracy clearly beats a `LogisticRegression` baseline (KNN captures the non-linear boundary). |
| Edge | single feature; `metric="manhattan"`; `n_neighbors == n_samples`. |
| Reference (`-m reference`) | `predict` (exact) and `predict_proba` (`rtol=1e-6`) match `sklearn.neighbors.KNeighborsClassifier(algorithm="brute", …)` for both `weights` and both `metric`s. |

Plus `examples/knn.py` — seeded, runnable: a `k` sweep on noisy moons
showing the accuracy / boundary-smoothness tradeoff, and `"uniform"` vs
`"distance"` weighting.

## References

- Cover & Hart (1967), "Nearest Neighbor Pattern Classification", *IEEE
  Transactions on Information Theory* 13(1).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd
  ed., §2.3.2 (nearest-neighbour methods) and §13.3.
- Bishop, *Pattern Recognition and Machine Learning*, §2.5.2.
