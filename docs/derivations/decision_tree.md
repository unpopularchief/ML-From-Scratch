# Decision tree (CART classifier)

The finalized derivation walkthrough for
`scratchgrad.tree.DecisionTreeClassifier`, written per `plan.md` §0.3
*before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and mirrors the `linear/`,
[`knn.md`](knn.md), and [`gaussian_nb.md`](gaussian_nb.md) walkthroughs.

A tree has no global objective and no gradient. Like KNN, this document
takes the place of a derivation by writing out the decision rule, the
impurity math a split is scored on, the greedy search, and why greedy —
the "derivation" analogue for a method fitted by recursive partitioning
rather than optimisation.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — numeric features only ($X$ is `float64`) |
| $y$ | $(n,)$ | class labels |
| $\mathcal{C}$ | $(K,)$ | `classes_` $= \operatorname{np.unique}(y)$, sorted |
| node $t$ | — | a subset $S_t \subseteq \{1, \dots, n\}$ of training indices |
| $p_{tk}$ | scalar | class-$k$ fraction in $t$: $p_{tk} = \frac{1}{\lvert S_t\rvert}\sum_{i \in S_t}\mathbb{1}[y_i = k]$ |
| $(j, \tau)$ | — | a split: feature $j$, threshold $\tau$; $x$ goes left iff $x_j \le \tau$ |
| $H(t)$ | scalar | impurity of node $t$ |

CART grows a **binary** tree. Every internal node tests $x_j \le \tau$;
leaves hold a class-frequency vector. No categorical splits, no multi-way
splits. Naturally multiclass (`classes_` has no size limit).

## 2. Impurity — what a split is scored on

There is no closed form. CART does **greedy recursive partitioning**: at
each node, take the split that most reduces impurity, then recurse into
each child.

**Impurity of a node** with class distribution $p_{t\cdot}$:

$$H_{\text{gini}}(t) = \sum_{k} p_{tk}(1 - p_{tk}) = 1 - \sum_k p_{tk}^2
\qquad
H_{\text{entropy}}(t) = -\sum_{k} p_{tk}\log_2 p_{tk}$$

with $0\log_2 0 := 0$. Gini is the probability of misclassifying a sample
drawn from $t$ if it is labelled by sampling from $t$'s own class
distribution. Entropy is in **bits** ($\log_2$) — scikit-learn's
convention, so parity is exact. Both are $0$ at a pure node and maximal
($\frac{K-1}{K}$ for gini, $\log_2 K$ for entropy) at the uniform
distribution.

**Impurity decrease of a split.** $(j, \tau)$ partitions $S_t$ into $S_L$
($x_j \le \tau$) and $S_R$ ($x_j > \tau$), of sizes $n_L, n_R$, with
$n_t = n_L + n_R$:

$$\Delta H(j, \tau) = H(S_t) - \left[\frac{n_L}{n_t}H(S_L)
  + \frac{n_R}{n_t}H(S_R)\right]$$

CART chooses $(j^\*, \tau^\*) = \arg\max_{j, \tau}\Delta H(j, \tau)$. Since
$H(S_t)$ is fixed at the node, this is the same as minimising the weighted
child impurity in the brackets. (When $H$ is entropy, $\Delta H$ is the
*information gain*.)

**The `min_impurity_decrease` gate** uses scikit-learn's total-weighted
form, so one threshold means the same thing at every depth:

$$\frac{n_t}{n}\,\Delta H(j^\*, \tau^\*) \ \ge\ \texttt{min\_impurity\_decrease}
  \quad\text{— otherwise the node becomes a leaf.}$$

## 3. Finding $\tau^\*$ for one feature

The weighted child impurity is piecewise-constant in $\tau$ between
consecutive distinct values of $x_j$, so the only thresholds worth
checking are **midpoints of adjacent distinct sorted values**. For feature
$j$:

1. sort the node's rows by $x_j$;
2. let the left partition be the first $m$ sorted rows, for $m = 1, \dots,
   n_t - 1$. Left class counts are the running cumulative sum of the
   one-hot labels; the right counts are (parent $-$ left);
3. for each $m$ where $v_{(m)} \ne v_{(m+1)}$ and both sides have at least
   `min_samples_leaf` rows, score the split at
   $\tau = \tfrac{1}{2}(v_{(m)} + v_{(m+1)})$ from those counts.

$O(d\,n_t\log n_t)$ per node (the sort dominates). The pseudocode in §6
shows this as a vectorised cumulative-count sweep; a brute-force
"recompute both child impurities from scratch at every candidate
threshold" version is $O(d\,n_t^2)$ and is what the tests check against.

**Ties.** The first split found wins: lowest feature index, then lowest
threshold. Deterministic — unlike scikit-learn, which permutes features
with its RNG and so breaks equal-gain ties randomly.

## 4. Recursion, stopping, leaves

```
build(S, depth):
    value    = class_counts(S) / |S|            # predicted distribution at this node
    if  depth == max_depth  or  |S| < min_samples_split  or  H(S) == 0:
        return Leaf(value)
    (j, τ, ΔH) = best_split(S)                   # skips any split making a child < min_samples_leaf
    if  j is None  or  (|S| / n) · ΔH < min_impurity_decrease:
        return Leaf(value)
    return Node(j, τ,
                left  = build({i ∈ S : X[i, j] ≤ τ}, depth + 1),
                right = build({i ∈ S : X[i, j] >  τ}, depth + 1))
```

**Stop → leaf** when any of: `depth == max_depth`; `|S| <
min_samples_split`; the node is pure ($H(S) = 0$); every feature is
constant within $S$ (no candidate split); no candidate leaves both
children with $\ge$ `min_samples_leaf` rows; the best split fails the
`min_impurity_decrease` gate.

**Leaf prediction.** `predict_proba` returns the leaf's `value` vector;
`predict` returns `classes_[argmax(value)]` (ties → lowest label, since
`classes_` is sorted).

## 5. Why greedy — and what it costs

Building the globally optimal tree is **NP-complete** (Hyafil & Rivest,
1976). The greedy top-down heuristic is what makes trees practical.
Documented consequences:

- A feature useful only *in interaction* (the classic case: XOR) can be
  passed over — every depth-1 split has $\Delta H \approx 0$. The tree
  still fits XOR (some split is taken anyway, deterministically, and the
  depth-2 children separate it), but greediness is a real limitation.
- An unbounded tree drives training impurity to $0$ and overfits.
  `max_depth`, `min_samples_leaf`, `min_samples_split`,
  `min_impurity_decrease` are the regularisers. **No cost-complexity
  pruning** (`ccp_alpha`) in this estimator — a documented omission, like
  KNN's missing KD-tree.

## 6. Algorithm (pseudocode)

```
DecisionTreeClassifier(criterion="gini", max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,
                       min_impurity_decrease=0.0)

_gini(counts):     p = counts / counts.sum();  return 1 - Σ p²
_entropy(counts):  p = counts / counts.sum();  return -Σ_{p>0} p·log2(p)

_best_split(X_node, y_idx, K, H, min_leaf):          # -> (feature|None, threshold, ΔH)
    parent = H(bincount(y_idx, K))
    best_gain, best_j, best_τ = 0, None, 0
    onehot = eye(K)[y_idx]
    for j in range(d):
        order    = argsort(X_node[:, j])
        x        = X_node[order, j]
        left     = cumsum(onehot[order], axis=0)     # (n_t, K): counts of first m+1 rows
        right    = parent_counts - left
        n_left   = arange(1, n_t);  n_right = n_t - n_left
        valid    = (x[:-1] != x[1:]) & (n_left >= min_leaf) & (n_right >= min_leaf)
        child    = (n_left·H(left[:-1]) + n_right·H(right[:-1])) / n_t
        gains    = where(valid, parent - child, -inf)
        m        = argmax(gains)
        if gains[m] > best_gain + 1e-12:             # strict -> first (j, τ) wins ties
            best_gain, best_j, best_τ = gains[m], j, (x[m] + x[m+1]) / 2
    return best_j, best_τ, best_gain

_build(X, y_idx, depth):
    counts   = bincount(y_idx, K);  value = counts / len(y_idx);  imp = H(counts)
    node     = Node(n_samples=len(y_idx), impurity=imp, value=value)
    if depth == max_depth or len(y_idx) < min_samples_split or imp == 0:
        return node                                  # leaf
    j, τ, gain = _best_split(X, y_idx, K, H, min_samples_leaf)
    if j is None or (len(y_idx) / n) · gain < min_impurity_decrease:
        return node                                  # leaf
    mask = X[:, j] <= τ
    node.feature, node.threshold = j, τ
    node.left  = _build(X[mask],  y_idx[mask],  depth + 1)
    node.right = _build(X[~mask], y_idx[~mask], depth + 1)
    feature_importances_[j] += (len(y_idx) / n) · gain
    return node

fit(X, y):
    validate hyperparameters
    X, y = check_X_y(X, y)
    classes_ = unique(y);  y_idx = searchsorted(classes_, y)
    tree_ = _build(X, y_idx, depth=0)
    feature_importances_ /= feature_importances_.sum()   # if any split was made

predict_proba(X):  descend each row to its leaf, return the leaf `value` rows
predict(X):        classes_[argmax(predict_proba(X), axis=1)]
score(X, y):       accuracy_score(y, predict(X))
```

Module-level `_gini`, `_entropy`, and `_best_split` are factored out so the
impurity math and the split search are unit-tested directly, mirroring the
`_*` helpers in `linear/`, `neighbors/`, and `naive_bayes/`.

## 7. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | `_gini` / `_entropy` on hand distributions: pure → 0; 50/50 → gini 0.5, entropy 1.0 bit; 3-way uniform → gini 2/3, entropy $\log_2 3$. Hand-laid 1-feature node: `_best_split` returns the known $(j, \tau)$ (midpoint) and $\Delta H$. |
| Split search | Tiny 2-D set with an obvious axis-aligned gap → root split is the correct $(j, \tau)$; `_best_split` against an independent $O(n^2)$ brute-force version on random nodes. |
| Independent reference | A dead-simple recursive builder (brute-force impurity at every candidate threshold, no sweep) agrees with `tree_` structure and `predict` on seeded data — the gradient-check analogue. |
| Determinism | Same data + params → structurally identical tree across runs. |
| Regularisers | Unbounded tree → 100% train accuracy on separable data; `max_depth=1` → a stump (`max_depth_ == 1`, both children leaves); large `min_impurity_decrease` / `min_samples_split > n` → root is a leaf; every leaf has $\ge$ `min_samples_leaf` rows. |
| proba | rows sum to 1; shape $(m, K)$; `predict == classes_[argmax(proba)]`. |
| feature_importances_ | sums to 1 when the tree splits; all-zero when the root is a leaf; concentrates on the informative feature when only one matters. |
| Contract | `NotFittedError` before `fit`; `fit` returns self; hyperparameters unmutated; feature-count mismatch in `predict` raises; bad `criterion`, `max_depth < 1`, `min_samples_split < 2`, `min_samples_leaf < 1`, `min_impurity_decrease < 0` raise; `repr` round-trips. |
| Behavioral | `make_moons(noise=0.2)`: a depth-capped tree beats a `LogisticRegression` baseline on held-out data. 3-class `make_blobs` → high held-out accuracy. |
| Edge | single feature; single sample (→ leaf); constant features (→ leaf); single-class `y` (→ leaf, impurity 0); duplicate rows with conflicting labels (→ mixed-distribution leaf). |
| Reference (`-m reference`) | `predict` (exact) and `predict_proba` (`rtol`) match `sklearn.tree.DecisionTreeClassifier` for both criteria across several `max_depth`, plus a `min_samples_leaf` case. `feature_importances_` is checked only approximately (`atol`): when two features tie for the best split, our "lowest index" rule and scikit-learn's RNG feature permutation can attribute that node's decrease to different features even though the predictions are identical. |

Plus `examples/decision_tree.py` — seeded: the train/test accuracy curve
across `max_depth` (the overfitting story), the fitted tree printed as
indented text, and `feature_importances_`.

## Scope — what this estimator omits (and why)

- **`DecisionTreeRegressor`** (MSE criterion, mean-value leaves) — a small
  follow-up one-file PR sharing this recursion, exactly as
  `KNeighborsRegressor` was deferred after the classifier.
- **`random_state` / `max_features`** — with deterministic tie-breaking and
  all features considered they would be no-ops. They return in M2 when
  `RandomForest` needs feature subsampling.
- **`ccp_alpha`** (cost-complexity pruning), **`class_weight`**,
  **`splitter`** (best only), **float `min_samples_*`** (fractions),
  **categorical features** — all standard scikit-learn options left out to
  keep the lesson (impurity + greedy recursion) uncluttered.

## References

- Breiman, Friedman, Olshen, Stone, *Classification and Regression Trees*
  (Wadsworth, 1984).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd
  ed., §9.2.
- Hyafil & Rivest (1976), "Constructing optimal binary decision trees is
  NP-complete", *Information Processing Letters* 5(1).
- scikit-learn User Guide, §1.10 (Decision Trees).
