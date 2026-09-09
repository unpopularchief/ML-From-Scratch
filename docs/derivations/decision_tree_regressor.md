# Decision tree (CART regressor)

The finalized derivation walkthrough for
`scratchgrad.tree.DecisionTreeRegressor`, written per `plan.md` §0.3
*before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and this document is the
regression companion to [`decision_tree.md`](decision_tree.md) — read that
one first. Everything about the *search* (greedy recursive partitioning,
midpoint thresholds, the deterministic tie-break, `max_features` /
`random_state` per-node subsampling, `sample_weight` threading) is
identical; only the **impurity function**, the **leaf value**, and the
**prediction / scoring head** change.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — numeric features only ($X$ is `float64`) |
| $y$ | $(n,)$ | **continuous** targets (`float64`); no `classes_` |
| node $t$ | — | a subset $S_t \subseteq \{1, \dots, n\}$ of training indices |
| $w$ | $(n,)$ | `sample_weight` $\in \mathbb{R}_{\ge 0}^n$ (default: all ones) |
| $N_t$ | scalar | node mass $\sum_{i \in S_t} w_i$; total mass $N = \sum_i w_i$ |
| $\bar y_t$ | scalar | weighted node mean $\frac{1}{N_t}\sum_{i \in S_t} w_i y_i$ |
| $(j, \tau)$ | — | a split: feature $j$, threshold $\tau$; $x$ goes left iff $x_j \le \tau$ |
| $H(t)$ | scalar | impurity of node $t$ (here: within-node variance of $y$) |

CART grows the same **binary** tree. Every internal node tests
$x_j \le \tau$; each leaf now holds a **single scalar** $\bar y_t$ rather
than a class-frequency vector.

## 2. Impurity — within-node variance

A regression tree scores a split on how much it reduces the **variance of
$y$** inside a node. With weighted mean $\bar y_t$:

$$H_{\mathrm{mse}}(t) \;=\; \frac{1}{N_t}\sum_{i \in S_t} w_i\,(y_i - \bar y_t)^2
  \;=\; \frac{\sum_{i \in S_t} w_i y_i^2}{N_t} \;-\; \bar y_t^{\,2}.$$

The second (sum-of-squares) form is what the sweep in §4 uses. $H = 0$
iff every $y_i$ in the node is identical — the regression analogue of a
*pure* node. This estimator implements this one criterion,
`criterion="squared_error"` (scikit-learn's current spelling);
`absolute_error` (median leaves, mean-absolute deviation), `friedman_mse`,
and `poisson` are out of scope — see the Scope note.

**Why the mean is the leaf value.** The constant $c$ minimising the
node's weighted squared error is exactly the weighted mean:

$$\frac{\partial}{\partial c}\sum_{i \in S_t} w_i (y_i - c)^2
  = -2\sum_{i \in S_t} w_i (y_i - c) = 0
  \quad\Longrightarrow\quad
  c^\* = \frac{\sum_i w_i y_i}{\sum_i w_i} = \bar y_t,$$

and the minimised value is $N_t\,H_{\mathrm{mse}}(t)$. So the node impurity
*is* the best achievable sum-of-squared-errors from a single constant, per
unit mass, and the leaf predicts that constant. This mirrors the
classifier, where the leaf's class-frequency vector is the minimiser of
the node's expected loss.

## 3. The split score is greedy training-SSE reduction

The impurity-decrease formula is **unchanged** from the classifier —
$(j, \tau)$ partitions $S_t$ into $S_L$ ($x_j \le \tau$) and $S_R$, with
masses $N_L, N_R$ and $N_t = N_L + N_R$:

$$\Delta H(j, \tau) = H(S_t) - \left[\frac{N_L}{N_t}H(S_L)
  + \frac{N_R}{N_t}H(S_R)\right].$$

**Law of total variance.** Decomposing the node variance around the child
means,

$$H(S_t) = \underbrace{\frac{N_L}{N_t}H(S_L) + \frac{N_R}{N_t}H(S_R)}_{\text{within-group}}
  \;+\; \underbrace{\frac{N_L}{N_t}(\bar y_L - \bar y_t)^2
    + \frac{N_R}{N_t}(\bar y_R - \bar y_t)^2}_{\text{between-group}},$$

so $\Delta H(j, \tau)$ is exactly the **between-group (explained)
variance** of the split, and is $\ge 0$ for every split — a split can
never increase the mass-weighted mean variance.

**Connection to training loss.** If every node predicts its own mean, the
total weighted training SSE is $\sum_{\text{leaves } t} N_t H(t)$.
Splitting node $t$ changes it by

$$N_t H(S_t) - N_L H(S_L) - N_R H(S_R) \;=\; N_t\,\Delta H(j, \tau) \;\ge\; 0,$$

so $\arg\max_{j,\tau}\Delta H$ is precisely the split that greedily
reduces training SSE the most. (This is algebraically scikit-learn's MSE
criterion: it maximises the proxy
$\frac{S_L^2}{W_L} + \frac{S_R^2}{W_R}$ where $S_\bullet = \sum w_i y_i$,
which equals $-N_L H(S_L) - N_R H(S_R)$ up to the node-constant
$\sum_{i \in S_t} w_i y_i^2$ — same $\arg\max$.)

**The `min_impurity_decrease` gate** uses the same total-weighted form as
the classifier, $\frac{N_t}{N}\,\Delta H(j^\*, \tau^\*) \ge
\texttt{min\_impurity\_decrease}$, and `feature_importances_` accumulates
$\frac{N_t}{N}\,\Delta H$ per split feature, renormalised to sum to 1.

## 4. Finding $\tau^\*$ for one feature

Same sweep as the classifier, with three scalar running sums replacing the
$K$-vector of cumulative one-hot counts. For feature $j$:

1. sort the node's rows by $x_j$;
2. let the left partition be the first $m$ sorted rows, $m = 1, \dots,
   n_t - 1$. Maintain the cumulative **weighted** sums
   $W_L = \sum w_i,\; P_L = \sum w_i y_i,\; Q_L = \sum w_i y_i^2$ over
   those rows; the right side is (parent $-$ left);
3. the variance of either side comes straight from its sums,
   $H = \dfrac{Q}{W} - \left(\dfrac{P}{W}\right)^2$ (clamped at $0$ —
   catastrophic cancellation can push it a few ulps negative);
4. for each $m$ where $v_{(m)} \ne v_{(m+1)}$ and both sides keep at least
   `min_samples_leaf` **rows**, score the split at
   $\tau = \tfrac12(v_{(m)} + v_{(m+1)})$.

$O(d\,n_t\log n_t)$ per node (the sort dominates); the tests check it
against an $O(d\,n_t^2)$ recompute-from-scratch brute force.

**Centre $y$ first, or the sum form is unusable.** Step 3 is the textbook
one-pass variance, and it subtracts two nearly equal numbers: when
$|\bar y| \gg$ the spread, $Q/W$ and $(P/W)^2$ agree to many digits and the
cancellation eats all of them. A spread-1 target offset by $10^9$ has
$Q/W \approx (P/W)^2 \approx 10^{18}$, where float64 resolves only to
$\approx 2\times10^{2}$ — the computed variance comes out **exactly
$0.0$**, the node looks pure, and the tree stops splitting. Targets with a
large baseline are ordinary (timestamps, prices, Kelvin), so this is not a
corner case.

Because variance is exactly shift-invariant, the cure is free: subtract the
node mean once, $y \mapsto y - \bar y_t$, before accumulating $P_L, Q_L$.
Every gain is unchanged mathematically, one extra $O(n_t)$ pass is spent,
and the offset-$10^9$ tree above goes from wrong to agreeing with the
uncentred-target tree to $\approx 10^{-7}$. scikit-learn does **not** do
this — its criterion accumulates raw $\sum y$ and $\sum y^2$ — so on
large-baseline targets its splits degrade exactly as ours did before the
centring. (The node impurity stored on `_Node` in §5 uses the centred
two-pass form $\frac{1}{N_t}\sum w_i(y_i - \bar y_t)^2$ for the same
reason, and so is exactly $0$ on a constant-$y$ node.)

**Ties.** Identical rule to the classifier: among splits within a tolerance
of the best gain, the lowest feature index wins, then the lowest
threshold — applied independently of the feature visit order, so the
drawn *subset* matters but not its permutation.

**The tolerance must be relative here, unlike the classifier.** A gini or
entropy decrease always lies in $[0, 1]$, so the classifier's absolute
`_MIN_GAIN` $=10^{-12}$ is a genuine "this is float noise" floor. A
*variance* decrease carries the units of $y^2$: rescaling $y \mapsto sy$
scales every gain by $s^2$ while leaving the best split unchanged, so an
absolute floor would reject real splits purely because the target is
small. With $y$ in millivolts, $\mathrm{Var}(y) \approx 2\times10^{-12}$
and **every** split falls under $10^{-12}$ — the root becomes a leaf and
the estimator silently degenerates to predicting the mean ($R^2 = 0$).

Since $\Delta H \le H(S_t)$ always, the fix is to scale the floor by the
node's own impurity,

$$\texttt{gain\_atol} = 10^{-12}\, H(S_t),$$

which asks the question the guard is actually for — *is this gain a
negligible fraction of this node's variance?* — and makes the fitted tree
invariant to the units of $y$. Float noise in $\Delta H$ is $O(\epsilon
H(S_t))$ with $\epsilon \approx 2.2\times10^{-16}$, so a relative floor of
$10^{-12}$ still sits four orders of magnitude above the noise. Note this
is *stricter* than scikit-learn, which stops on an absolute
`impurity <= np.finfo(np.float64).eps` and so collapses to a single leaf
once $\mathrm{Var}(y) \lesssim 2\times10^{-16}$; our tree does not.

## 4a. Feature subsampling and sample weights

Both mechanisms are **exactly** as in
[`decision_tree.md`](decision_tree.md) §3a / §3b:

- **`max_features` / `random_state`** resolve and draw per node the same
  way (`None`/`"sqrt"`/`"log2"`/`int`/`float`, `max()`-clamped to $\ge 1$;
  constant-in-node features skipped for free; the walk stops after $m$
  non-constant features scored). `max_features=None` *and*
  `random_state=None` ⇒ no permutation drawn, index-order sweep, and the
  tree is byte-identical to the unseeded fit.
- **`sample_weight`** scales every sum above; the `min_samples_*` gates
  still count **rows**, not mass (`min_weight_fraction_leaf` is omitted);
  **zero-weight rows are dropped before fitting**, as scikit-learn does.
  `None` ≡ all-ones ≡ the unweighted fit; integer weights ≡ a fit on the
  row-duplicated dataset.

**Parity with scikit-learn.** On the deterministic path
(`max_features=None`, uniform or shallow weights) *and while nodes stay
reasonably large*, the split at each node is unique and `predict` matches
`sklearn.tree.DecisionTreeRegressor` **exactly**, to floating-point
round-off.

The limit on that is **node size, not depth**, and it bites harder here
than one might expect. Continuous, noise-free $y$ makes exact gain ties
look measure-zero — but $\Delta H$ depends only on the *partition*, not on
which feature produced it. Once a node is down to a handful of rows, its
best split usually just isolates one sample, and *several* features will
each cut that same single sample off, so their gains agree **to the last
bit**. On the reference fixtures this floor is reached at depth 6: at a
4-row node, features 1, 2 and 3 all score $0.03889486503748918$. We take
the lowest index; scikit-learn draws features in a random permutation —
which it does *even when* `max_features` is `None` — and keeps the first
one it finds. So a fully-grown tree diverges from scikit-learn's in
*structure*, while remaining an equally valid greedy CART tree: same
depth, same leaf count, same exact interpolation of the training set,
held-out $R^2$ within noise.

Riding along with that is a smaller, purely numerical gap: **scikit-learn
casts `X` to `float32` internally**, so its thresholds are float32
midpoints where ours are float64 (at the root, $2.0692250728607178$ versus
$2.0692250801282026$). That leaves a $\sim 10^{-8}$ band around every
boundary in which a query row can be routed to the other child. Harmless
in a shallow tree; in a 17-level one there are enough boundaries for it to
matter.

The same divergence appears whenever `max_features < d`, or deep down a
heavily reweighted tree. All of those cases are comparable only by
held-out $R^2$ — and for `max_features`, only *averaged over seeds*: a
single subsampled depth-6 tree on 3-of-10 features has an $R^2$ standard
deviation of about $0.18$ across seeds, far larger than any systematic
difference between the two implementations. `feature_importances_` is
likewise checked only approximately (`atol`), for the same
equal-gain-attribution reason as the classifier.

## 5. Recursion, stopping, leaves

```
build(S, depth):
    value = weighted_mean(y[S], w[S])           # the leaf's scalar prediction
    if  depth == max_depth  or  |S| < min_samples_split  or  H(S) == 0:
        return Leaf(value)
    (j, τ, ΔH) = best_split(S)                   # skips splits leaving a child < min_samples_leaf rows
    if  j is None  or  (N_S / N) · ΔH < min_impurity_decrease:
        return Leaf(value)
    return Node(j, τ,
                left  = build({i ∈ S : X[i, j] ≤ τ}, depth + 1),
                right = build({i ∈ S : X[i, j] >  τ}, depth + 1))
```

**Stop → leaf** when any of: `depth == max_depth`; `|S| <
min_samples_split` (rows); the node is pure ($H(S) = 0$, i.e. constant
$y$); every candidate feature is constant within $S$; no candidate leaves
both children with $\ge$ `min_samples_leaf` rows; the best split fails the
`min_impurity_decrease` gate.

**Prediction.** `predict(X)` descends each row to its leaf and returns the
leaf's scalar `value`; shape $(n,)$. There is no `predict_proba`.
`score(X, y)` returns the coefficient of determination
$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$
(via `scratchgrad.metrics.r2_score`), matching the regressor convention
already used by `LinearRegression`.

## 6. Algorithm (pseudocode)

```
DecisionTreeRegressor(criterion="squared_error", max_depth=None,
                      min_samples_split=2, min_samples_leaf=1,
                      min_impurity_decrease=0.0,
                      max_features=None, random_state=None)

_variance_from_sums(W, P, Q):   return max(0, Q/W - (P/W)²)   # weighted variance

_best_split(X_node, y, min_leaf, m, rng, w):     # -> (feature|None, threshold, ΔH)
    W, P, Q  = w.sum(), (w·y).sum(), (w·y²).sum()
    parent   = _variance_from_sums(W, P, Q)
    best_gain, best_j, best_τ = 0, None, 0
    order_of_j = range(d) if rng is None else rng.permutation(d)
    scored = 0
    for j in order_of_j:
        order    = argsort(X_node[:, j])
        x        = X_node[order, j]
        distinct = x[:-1] != x[1:]
        if not distinct.any():  continue            # constant in node -> no budget used
        W_L = cumsum(w[order])[:-1];  P_L = cumsum((w·y)[order])[:-1];  Q_L = cumsum((w·y²)[order])[:-1]
        W_R, P_R, Q_R = W - W_L, P - P_L, Q - Q_L
        n_left  = arange(1, n_t);  n_right = n_t - n_left
        valid   = distinct & (n_left >= min_leaf) & (n_right >= min_leaf)
        child   = (W_L·_variance_from_sums(W_L, P_L, Q_L)
                 + W_R·_variance_from_sums(W_R, P_R, Q_R)) / W
        gains   = where(valid, parent - child, -inf)
        i       = argmax(gains)
        if gains[i] > 1e-12 and (gains[i] > best_gain + 1e-12
                                 or (best_j is not None
                                     and gains[i] > best_gain - 1e-12 and j < best_j)):
            best_gain, best_j, best_τ = gains[i], j, (x[i] + x[i+1]) / 2
        scored += 1
        if m is not None and scored >= m:  break
    return best_j, best_τ, best_gain

_build(X, y, w, depth):
    W      = w.sum();  mean = (w·y).sum() / W
    imp    = weighted_mean(w, (y - mean)²)          # centred form -> exactly 0 on constant y
    node   = Node(n_samples=len(y), impurity=imp, value=mean)
    if depth == max_depth or len(y) < min_samples_split or imp == 0:
        return node                                  # leaf
    j, τ, gain = _best_split(X, y, min_samples_leaf, m, rng, w)
    if j is None or (W / N) · gain < min_impurity_decrease:
        return node                                  # leaf
    mask = X[:, j] <= τ
    node.feature, node.threshold = j, τ
    node.left  = _build(X[mask],  y[mask],  w[mask],  depth + 1)
    node.right = _build(X[~mask], y[~mask], w[~mask], depth + 1)
    feature_importances_[j] += (W / N) · gain
    return node

fit(X, y, sample_weight=None):
    validate hyperparameters                         # same checks as the classifier
    X, y = check_X_y(X, y)
    w   = check_sample_weight(sample_weight, n)       # None -> ones
    keep = w > 0;  X, y, w = X[keep], y[keep], w[keep]    # drop zero-weight rows
    m   = _resolve_max_features(max_features, d)      # reused from decision_tree.py
    rng = None if (max_features is None and random_state is None) \
               else check_random_state(random_state)
    tree_ = _build(X, y, w, depth=0)
    feature_importances_ /= feature_importances_.sum()    # if any split was made

predict(X):  descend each row to its leaf, return the leaf scalar `value`
score(X, y): r2_score(y, predict(X))
```

`_Node`, `_leaf_for`, and `_depth` are imported from
`scratchgrad.tree.decision_tree` unchanged (`_Node.value` just widens to
`FloatArray | float`). `_variance_from_sums` and `_best_split` are
module-level so the impurity math and the split search are unit-tested
directly, mirroring `_gini` / `_entropy` / `_best_split` in the classifier.

## 7. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | `_variance_from_sums` on hand values (constant → 0; $\{0, 2\}$ equal-weight → 1; a weighted pair → the weighted variance). The node mean minimises SSE (numerically, vs a grid of constants). Hand-laid 1-feature node: `_best_split` returns the known $(j, \tau)$ midpoint and $\Delta H$ equal to the between-group variance. |
| Split search | Tiny set with an obvious axis-aligned step in $y$ → root split is the correct $(j, \tau)$; `_best_split` against an independent $O(n^2)$ brute force on random nodes (uniform and weighted). |
| Independent reference | A dead-simple recursive builder (brute-force variance at every candidate threshold, no sweep) agrees with `tree_` structure and `predict` on seeded data — the gradient-check analogue. |
| Determinism | Same data + params → structurally identical tree across runs. Seeded (`max_features` + `random_state`): same seed → identical, different seed → generally different; `max_features=None, random_state=None` unchanged from the unseeded path. |
| Sample weights | `sample_weight=None` ≡ all-ones ≡ the unweighted tree. Integer weights ≡ a fit on the row-duplicated dataset (structure, `predict`, `feature_importances_`). Zero-weight rows ≡ dropped rows. A conflicting-value pair with unequal weights → leaf `value` is the weighted mean. |
| Feature subsampling | Every realised split is on one of that node's $m$ drawn features; `max_features=1` still fits a separable step target given enough depth; bad `max_features` raises (validator reused from the classifier). |
| Regularisers | Unbounded tree → $R^2 = 1$ on a noise-free piecewise-constant target; `max_depth=1` → a stump (`max_depth_ == 1`, both children leaves); large `min_impurity_decrease` / `min_samples_split > n` → root is a leaf (all-zero `feature_importances_`); every leaf has $\ge$ `min_samples_leaf` rows. |
| Target scaling | Rescaling $y$ by $s \in [10^{-9}, 10^{6}]$ leaves the tree structurally identical (same leaf count, same depth) and every prediction exactly rescaled — the relative gain floor of §4. Explicit regression guard at $s = 10^{-6}, 10^{-9}$: the root must still split and reach $R^2 > 0.9$ (an absolute floor silently collapsed this to one leaf at $R^2 = 0$). |
| Target offset | Shifting $y$ by $c \in \{10^3, 10^6, 10^9\}$ leaves the leaf count unchanged and every prediction shifted by exactly $c$ — the §4 centring. Without it, $c = 10^9$ drove the computed variance to $0.0$ and flattened the tree. |
| feature_importances_ | Sums to 1 when the tree splits; all-zero when the root is a leaf; concentrates on the informative feature when only one drives $y$. |
| Contract | `NotFittedError` before `fit`; `fit` returns self; hyperparameters unmutated; feature-count mismatch in `predict` raises; bad `criterion`, `max_depth < 1`, `min_samples_split < 2`, `min_samples_leaf < 1`, `min_impurity_decrease < 0` raise; `score` is $R^2$; `repr` round-trips; `predict` output shape $(n,)$. |
| Behavioral | A nonlinear target ($y = \sin 3x_0 + x_1^2 + \varepsilon$): a depth-capped tree beats a `LinearRegression` baseline on held-out $R^2$. A 1-D step function is recovered near-exactly. |
| Edge | single feature; single sample (→ leaf, predicts that value); constant $y$ (→ leaf, impurity 0); constant $X$ (→ leaf, predicts the mean); duplicate rows with conflicting $y$ (→ leaf at their mean). |
| Reference (`-m reference`) | On the deterministic path *with large nodes*: `predict` (exact) matches `sklearn.tree.DecisionTreeRegressor` at `max_depth` 1/3/5, and for a `min_samples_leaf=8` tree (the gate keeps nodes big, so exactness survives to full depth). A *fully grown* tree instead asserts what is well defined once tiny nodes make features tie bit-for-bit: exact interpolation of the training set, equal depth, equal leaf count, held-out $R^2$ within $0.05$. `feature_importances_` is `atol`-tolerant (equal-gain attribution, as for the classifier). A *shallow* weighted tree matches `predict` exactly; a *deep* weighted tree is held-out-$R^2$ tolerance only, and the `max_features` tree compares mean $R^2$ **over 10 seeds** (one seed is pure noise — see §4a). |

Plus `examples/decision_tree_regressor.py` — seeded: the train/test $R^2$
curve across `max_depth` (the overfitting story) on a noisy sine, the
fitted tree printed as indented text, `feature_importances_`, and a 1-D
step-function fit shown as a small table.

## Scope — what this estimator omits (and why)

- **`criterion="absolute_error"`** (MAE): leaf = weighted median, impurity
  = mean absolute deviation. There is no cumulative-sum trick — the median
  must be recomputed as the split point moves, an $O(n_t^2\log n_t)$ inner
  loop — and it buys only outlier-robustness, not a new idea. Left out,
  like the classifier left out `ccp_alpha`.
- **`friedman_mse`** (Friedman's split-improvement score, scikit-learn's
  default and what `GradientBoosting` uses for *ranking* splits) and
  **`poisson`** — both variations on the same variance-reduction theme,
  omitted to keep one clean criterion.
- **`ccp_alpha`** (cost-complexity pruning), **`min_weight_fraction_leaf`**,
  **`splitter="random"`**, **float `min_samples_*`**, **categorical
  features** — all standard scikit-learn options left out, exactly as in
  the classifier.

## References

- Breiman, Friedman, Olshen, Stone, *Classification and Regression Trees*
  (Wadsworth, 1984) — §8 (regression trees).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd
  ed., §9.2.2.
- scikit-learn User Guide, §1.10 (Decision Trees); `DecisionTreeRegressor`.
