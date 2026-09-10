# Gradient boosting (classification, log-loss / deviance)

The finalized walkthrough for `scratchgrad.ensemble.GradientBoostingClassifier`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md); it builds on
[`decision_tree_regressor.md`](decision_tree_regressor.md) (every round fits
one) and shares the log-loss identity with
[`logistic_regression.md`](logistic_regression.md).

Where AdaBoost minimises an exponential loss by *reweighting* rows,
gradient boosting (Friedman, 2001) does **gradient descent in function
space**: it keeps a raw score function, and each round adds the regression
tree that best fits the negative gradient of the loss evaluated at the
current scores. The leaf values are then replaced by a one-step Newton
estimate of the loss-minimising constant on each leaf ("TreeBoost").

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$, $y$ | $(n, d)$, $(n,)$ | design matrix, class labels |
| $\mathcal{C}$ | $(K,)$ | `classes_` $= \mathrm{np.unique}(y)$, sorted |
| $K$ | scalar | `n_classes_` |
| $Y$ | $(n, K)$ | one-hot targets, $Y_{ik} = \mathbb{1}[y_i = \mathcal{C}_k]$ (binary: a single column $Y_{i} = \mathbb{1}[y_i = \mathcal{C}_1]$) |
| $M$ | scalar | `n_estimators` — rounds |
| $\nu$ | scalar | `learning_rate` (shrinkage), default $0.1$ |
| $F^{(m)}$ | $(n, S)$ | raw additive scores after round $m$; $S = 1$ for $K = 2$, else $S = K$ |
| $p^{(m)}$ | $(n, S)$ | probabilities: $\sigma(F)$ ($K=2$) or $\mathrm{softmax}(F)$ |
| $r^{(m)}$ | $(n, S)$ | pseudo-residuals — the negative loss gradient w.r.t. $F$ |
| $h_{mk}$ | — | the `DecisionTreeRegressor` fit to residual column $k$ in round $m$ |
| $R_{mkj}$ | — | region (leaf) $j$ of $h_{mk}$; $\gamma_{mkj}$ its refined value |
| $w_i$ | $(n,)$ | `sample_weight` (all ones by default) |

Naturally multiclass. Numeric features only (inherited from the base
tree). The base learner is **always** a `DecisionTreeRegressor` under
`criterion="squared_error"`; its `max_depth` (default $3$),
`min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` and
`max_features` are exposed, in the same spirit as AdaBoost exposing only
`max_depth`. There is no general `estimator=` / `init=` object.

## 2. Binary case — binomial deviance

One score $F(x)$; the positive-class probability is $p(x) = \sigma(F(x))$.
The per-sample loss is the negative Bernoulli log-likelihood, exactly the
form derived in [`logistic_regression.md`](logistic_regression.md) §2 (with
$y \in \{0, 1\}$ the indicator of $\mathcal{C}_1$):

$$L(y, F) = \mathrm{softplus}(F) - yF
  = -\big[y\log p + (1 - y)\log(1 - p)\big].$$

**Pseudo-residual** — the negative gradient w.r.t. the score, the quantity
the next tree is fit to:

$$r_i = -\left.\frac{\partial L}{\partial F}\right|_{F^{(m-1)}(x_i)}
      = y_i - \sigma\!\big(F^{(m-1)}(x_i)\big) = y_i - p_i .$$

**Round $m$:**

1. $p_i = \sigma(F^{(m-1)}(x_i))$; residuals $r_i = y_i - p_i$.
2. Fit a `DecisionTreeRegressor` $h_m$ to $(X, r)$ under `squared_error`.
   A least-squares fit to the negative gradient is the projection of the
   steepest-descent direction onto the space of trees — the "gradient" in
   gradient boosting.
3. **Leaf refinement (Friedman TreeBoost).** Keep the tree's *partition*
   but discard its mean-residual leaf values; replace each leaf $R_{mj}$
   with the constant that best reduces the *actual* loss on that leaf. One
   Newton step from $0$ — numerator $\sum(-L') = \sum r_i$, denominator
   $\sum L'' = \sum p_i(1 - p_i)$:

   $$\gamma_{mj} = \frac{\sum_{i \in R_{mj}} w_i\, r_i}
                        {\sum_{i \in R_{mj}} w_i\, p_i(1 - p_i)}
   \qquad(\gamma_{mj} = 0 \text{ if the denominator underflows}).$$

   A plain gradient step (leaf $=$ mean residual) is also a valid descent
   direction but converges far slower and does not match scikit-learn.
4. Update every row:
   $F^{(m)}(x_i) = F^{(m-1)}(x_i) + \nu\,\gamma_{m,\,j(x_i)}$. In code the
   fitted tree's leaf `value`s are overwritten with the $\gamma_{mj}$ and
   the update is `F += learning_rate * h_m.predict(X)` — scikit-learn's
   in-place trick.

**Initialisation.** $F^{(0)}(x) \equiv \log\dfrac{\bar y_w}{1 - \bar y_w}$,
the log-odds of the (weighted) base rate — the constant score minimising
the loss (`init="prior"`). `init="zero"` starts at $F^{(0)} \equiv 0$.

## 3. Multiclass case — multinomial deviance

Maintain $K$ scores $F_k(x)$; $p_k(x) = \mathrm{softmax}(F(x))_k$. With
one-hot $Y_{ik}$,

$$L(Y_i, F) = -\sum_k Y_{ik}\log p_{ik}, \qquad
  r_{ik} = -\frac{\partial L}{\partial F_k} = Y_{ik} - p_{ik}.$$

Each round fits **$K$ regression trees**, tree $k$ to residual column
$r_{\cdot k}$. Friedman's multinomial leaf update carries a $\frac{K-1}{K}$
factor (ESL eq. 10.29):

$$\gamma_{mkj} = \frac{K - 1}{K}\cdot
  \frac{\sum_{i \in R_{mkj}} w_i\, r_{ik}}
       {\sum_{i \in R_{mkj}} w_i\, |r_{ik}|\,(1 - |r_{ik}|)} .$$

For one-hot targets $|r_{ik}|(1 - |r_{ik}|) = p_{ik}(1 - p_{ik})$ exactly
(if $Y_{ik} = 1$ then $r = 1 - p \ge 0$; if $Y_{ik} = 0$ then $r = -p$),
so the denominator is the diagonal Hessian either way. Update
$F_{mk} \mathrel{+}= \nu\,\gamma_{mkj}$ per class.

**Initialisation.** $F^{(0)}_k = \log(\pi_k)$ with $\pi_k$ the weighted
class prior; $\mathrm{softmax}$ of that recovers the priors. The softmax
parameterisation is overcomplete (scores need not sum to zero) — harmless
and matches scikit-learn. $K = 2$ reduces to §2 up to that
reparameterisation, which is why the estimator uses the single-score
binomial path whenever `n_classes_ == 2`.

## 4. Stochastic gradient boosting (`subsample < 1`)

Each round draws a fraction `subsample` of the rows **without replacement**
and fits $h_{mk}$ on that subsample only; the leaf $\gamma_{mkj}$ also use
only the in-bag rows, while the score update $F^{(m)}$ is applied to every
row. This injects variance-reducing randomness (Friedman, 2002) and needs
`random_state`. `subsample = 1.0` (the default) is ordinary deterministic
gradient boosting.

`random_state` seeds one `numpy.random.Generator`; it drives the
`subsample` draw and, when `max_features` is set, is passed on as each
base tree's `random_state` (the per-node feature draw). When
`max_features is None` the trees get `random_state=None` and stay
byte-identical to a plain `DecisionTreeRegressor`.

## 5. The two scale gotchas from `decision_tree_regressor.md` apply here

Residual variance shrinks every round as the fit tightens. The base
tree's gain floor is **relative** to the node impurity
(`gain_atol = _MIN_GAIN * parent_impurity`, `decision_tree_regressor.md`
§4), which is exactly what keeps late-stage trees splitting instead of
collapsing to a stump once the residuals are small; and the tree centres
its moment sums per node, so an offset in the residuals (there is none
here, but the init score shifts $F$, not $r$) cannot cause cancellation.
Both already live in `DecisionTreeRegressor` — **it is reused unchanged**,
no source edit to `scratchgrad.tree`.

## 6. Prediction

Let $F(x)$ be the full additive score $F^{(0)} + \nu\sum_{m,k}\gamma$.

- **`decision_function(X)`** → $F(x)$: shape $(n,)$ for $K = 2$ (the single
  score, scikit-learn's sign convention — positive favours
  $\mathcal{C}_1$), $(n, K)$ otherwise.
- **`predict_proba(X)`** → $[\,1 - p,\; p\,]$ with $p = \sigma(F)$ for
  $K = 2$; $\mathrm{softmax}(F)$ otherwise. Rows sum to 1, columns follow
  `classes_`. Unlike AdaBoost's monotone score-calibration, this **is** a
  proper posterior estimate — deviance boosting targets the log-odds
  directly.
- **`predict(X)`** → threshold $p$ at $0.5$ ($K = 2$) / $\arg\max_k F_k$;
  ties resolve to the lowest class label.
- **`staged_predict(X)` / `staged_predict_proba(X)`** → the prediction /
  posterior of the partial ensemble after each round — the boosting curve.
- **`feature_importances_`** → mean over all fitted trees of the per-tree
  variance-decrease importances, renormalised to sum to 1.
- **`train_score_`** → the (weighted mean) training loss after each round,
  shape $(M,)$ — the empirical deviance curve.

## 7. What it costs, and scikit-learn parity

- **Fitting** is $M$ (binary) or $MK$ (multiclass) sequential tree builds;
  each round depends on the previous round's scores, so there is no
  parallelism across rounds and **no `n_jobs`**. A depth-3 tree build is
  $O(d\,n\log n)$.
- **No exact scikit-learn parity.** `GradientBoostingClassifier` defaults
  to `criterion="friedman_mse"` and searches splits over an RNG-permuted
  feature order; our base tree uses `squared_error` and the deterministic
  lowest-index tie-break. So the reference tests compare held-out
  accuracy and held-out log loss within a tolerance, and the *mean*
  absolute `predict_proba` difference — a handful of rows land in a
  differently-confident leaf and diverge, but the average posterior tracks
  closely — like the `RandomForest` and deep-tree references, not the
  tight `AdaBoost` one. `n_estimators=1, learning_rate=1, init="zero",
  max_depth=1` gives a tight analytic check against one hand-built Newton
  step.

## 8. Scope — what this estimator omits (and why)

- **Loss:** log-loss / deviance only. `loss="exponential"` (which makes GB
  equivalent to AdaBoost) is not implemented — AdaBoost already is.
- **`HistGradientBoostingClassifier`** (feature binning) — a different
  estimator, out of scope.
- **Early stopping** — `validation_fraction`, `n_iter_no_change`, `tol`:
  not implemented; `n_estimators` rounds always run.
- **`init` as an estimator object** — only the `"prior"` / `"zero"`
  strings.
- **`ccp_alpha`, `warm_start`, `max_leaf_nodes`, `n_jobs`,
  `GradientBoostingRegressor`** — deferred / out of scope.

## 9. Algorithm (pseudocode)

```
GradientBoostingClassifier(
    learning_rate=0.1, n_estimators=100, subsample=1.0,
    max_depth=3, min_samples_split=2, min_samples_leaf=1,
    min_impurity_decrease=0.0, max_features=None,
    init="prior", random_state=None)

_log_odds(y01, w):                 return log( wmean(y01) / (1 - wmean(y01)) )   # clipped
_prior_logits(Y_onehot, w):        return log( (w @ Y_onehot) / w.sum() )        # clipped, (K,)
_binary_negative_gradient(y01, F): return y01 - sigmoid(F)
_multinomial_negative_gradient(Y, F): return Y - softmax(F, axis=1)
_pseudo_hessian(g):                return |g| * (1 - |g|)          # = p_k(1-p_k) for one-hot
_newton_leaf_value(g, h, w, factor):
    den = sum(w * h)
    return 0.0 if den < 1e-12 else factor * sum(w * g) / den

fit(X, y, sample_weight=None):
    validate: n_estimators >= 1; learning_rate > 0; 0 < subsample <= 1;
              max_depth >= 1; init in {"prior", "zero"}
    X, y = check_X_y(X, y);  w = check_sample_weight(sample_weight, n)
    classes_ = unique(y);  K = len(classes_);  if K < 2: raise
    binary = (K == 2);  S = 1 if binary else K
    Y = onehot(y, classes_)                      # (n, S), single column if binary
    factor = 1.0 if binary else (K - 1) / K

    init_score_ = 0                if init == "zero"
                = _log_odds(...)   if binary
                = _prior_logits(...) otherwise
    F = tile(init_score_, (n, 1))                # (n, S)

    rng = Generator(random_state) if (random_state or subsample < 1 or max_features) else None
    tree_rng = rng if max_features is not None else None
    n_sub = round(subsample * n)

    estimators_ = [];  train_score = []
    for m in range(n_estimators):
        r = _binary_negative_gradient(Y[:,0], F[:,0])[:,None] if binary
            else _multinomial_negative_gradient(Y, F)
        hess = _pseudo_hessian(r)
        idx  = rng.choice(n, n_sub, replace=False) if subsample < 1 else arange(n)

        round_trees = []
        for k in range(S):
            tree = DecisionTreeRegressor(max_depth=..., ..., random_state=tree_rng)
            tree.fit(X[idx], r[idx, k], sample_weight=w[idx])
            for leaf in tree:                     # in-bag rows only
                leaf.value = _newton_leaf_value(r[members], hess[members], w[members], factor)
            F[:, k] += learning_rate * tree.predict(X)
            round_trees.append(tree)
        estimators_.append(round_trees)
        train_score.append(weighted_mean_deviance(Y, F, w))

    train_score_ = array(train_score)
    feature_importances_ = normalise(mean(t.feature_importances_ for t in all trees))
    return self

_raw_predict(X):   F = tile(init_score_, (m, 1));  for rt in estimators_: for k,t: F[:,k] += lr*t.predict(X);  return F
decision_function(X): F = _raw_predict(X);  return F[:,0] if binary else F
predict_proba(X):     F = _raw_predict(X);  return [1-σ(F0), σ(F0)] if binary else softmax(F)
predict(X):           classes_[argmax(predict_proba(X), axis=1)]
staged_*(X):          same, yielding after each round
score(X, y):          accuracy_score(y, predict(X))
```

Module-level `_log_odds`, `_prior_logits`, `_binary_negative_gradient`,
`_multinomial_negative_gradient`, `_pseudo_hessian`, `_newton_leaf_value`
are factored out and unit-tested directly, mirroring the `_*` helpers in
`linear/`, `tree/`, and `ensemble/`.

## 10. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | `_log_odds` on a hand example equals $\log\frac{\bar y}{1-\bar y}$; `_prior_logits` softmaxes back to the class frequencies. `_pseudo_hessian` equals $p(1-p)$ on one-hot residuals. `_newton_leaf_value` on hand values, and its $0$ guard when the denominator underflows. |
| Derivation stand-in (no differentiable objective *of the ensemble*) | one round with `init="zero"`, `learning_rate=1`, `max_depth=1`: the leaf values equal an independent Newton computation $\sum r / \sum p(1-p)$ per leaf, and $F^{(1)}$ equals that applied to every row; the residuals fed to round 2 equal $y - \sigma(F^{(1)})$ recomputed from scratch. |
| Negative gradient | `_binary_negative_gradient` / `_multinomial_negative_gradient` equal $Y - p$ for $p = \sigma(F)$ / $\mathrm{softmax}(F)$ on random $F$. |
| Monotone training loss | `train_score_` is non-increasing across rounds on separable-ish data (small `learning_rate`), for binary and 3-class. |
| Boosting behaviour | on `make_moons(noise=0.3)` the ensemble beats a single depth-3 tree on held-out data by a clear margin; `staged_predict` accuracy trends upward. |
| Shrinkage | smaller `learning_rate` with proportionally more `n_estimators` reaches comparable held-out accuracy; `learning_rate=1` overfits training faster than `0.1`. |
| Subsample | `subsample=0.5` is reproducible under a fixed `random_state` and differs from `subsample=1.0`; still fits (held-out accuracy above the majority baseline). |
| `init` | `init="zero"` sets `init_score_` to zeros; `init="prior"` sets it to the (weighted) log-odds / log-priors; both fit. |
| `predict_proba` / `decision_function` | shapes ($(n,)$ binary / $(n, K)$ multiclass for `decision_function`; always $(n, K)$ for `predict_proba`); rows of `predict_proba` sum to 1 and lie in $[0, 1]$; `predict == argmax`; binary `decision_function` sign agrees with `predict`. |
| `sample_weight` | passing integer `sample_weight` matches a fit on the row-duplicated data (deterministic path); a zero-weight row does not change the fit. |
| `feature_importances_` | shape $(d,)$, sums to 1, near-zero for a pure-noise feature. |
| `staged_predict` / `staged_predict_proba` | yield `n_estimators` items; the last equals `predict` / `predict_proba`. |
| Determinism | same data + `random_state` → identical `decision_function`. |
| Contract | `NotFittedError` before `fit`; `fit` returns self; hyperparameters unmutated; feature-count mismatch raises; `n_estimators < 1`, `learning_rate <= 0`, `subsample` outside $(0, 1]$, `max_depth < 1`, unknown `init`, single-class `y`, bad `sample_weight` all raise; `repr` round-trips. |
| Reference (`-m reference`) | vs `sklearn.ensemble.GradientBoostingClassifier`, binary and 3-class, `learning_rate` $\in \{0.1, 0.5\}$: held-out accuracy within `0.05`, held-out log loss within `0.1`, mean absolute `predict_proba` difference within `0.02`; `train_score_` shape matches. A `subsample=0.5` run also tracked to a held-out-accuracy tolerance. |

Plus `examples/gradient_boosting.py` — seeded, runnable: the staged
test-accuracy and `train_score_` curves, a `learning_rate` sweep, and a
single tree vs. the boosted ensemble.

## References

- Friedman (2001), "Greedy function approximation: a gradient boosting
  machine", *Annals of Statistics* 29(5) — the algorithm and TreeBoost.
- Friedman (2002), "Stochastic gradient boosting", *Computational
  Statistics & Data Analysis* 38(4) — the `subsample` variant.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §10 (Boosting and Additive Trees) — eq. 10.29 for the
  multinomial leaf update.
- scikit-learn User Guide, §1.11.4 (Gradient Tree Boosting).
