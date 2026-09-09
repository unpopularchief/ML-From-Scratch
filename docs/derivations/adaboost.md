# AdaBoost (SAMME, classification)

The finalized walkthrough for `scratchgrad.ensemble.AdaBoostClassifier`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and builds on
[`decision_tree.md`](decision_tree.md) (the weak learner) and its new §3b
(`sample_weight`, the reweighting hook).

Unlike a random forest, AdaBoost **does** have a derivation: it is forward
stagewise additive modelling of a **multi-class exponential loss**, and
every quantity in the algorithm — the weighted error, the estimator weight
$\alpha_m$, the sample-weight update — drops out of that one optimisation.
SAMME (Zhu, Zou, Rosset & Hastie, 2009) is the multi-class generalisation
of Freund & Schapire's original binary AdaBoost; it is scikit-learn's only
boosting algorithm since 1.6 (`SAMME.R` was removed).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$, $y$ | $(n, d)$, $(n,)$ | training design matrix, class labels |
| $c_i$ | scalar | the class of row $i$, in $\{1, \dots, K\}$ |
| $\mathcal{C}$ | $(K,)$ | `classes_` $= \mathrm{np.unique}(y)$, sorted |
| $M$ | scalar | `n_estimators` |
| $T_m$ | — | the depth-`max_depth` `DecisionTreeClassifier` fit in round $m$ (a stump when `max_depth=1`) |
| $w_i^{(m)}$ | $(n,)$ | sample weight entering round $m$, normalised so $\sum_i w_i = 1$ |
| $\mathrm{err}_m$ | scalar | weighted misclassification rate of $T_m$ — `estimator_errors_[m]` |
| $\alpha_m$ | scalar | additive weight of $T_m$ in the vote — `estimator_weights_[m]` |
| $\eta$ | scalar | `learning_rate` (shrinkage), default $1$ |

Naturally multiclass. Numeric features only (inherited from the base
tree). The base learner is **always** a `DecisionTreeClassifier`; only its
`max_depth` is exposed (default $1$ — a decision stump). There is no
general `estimator=` parameter and no `clone` machinery, in the same
spirit as KNN having no `algorithm` and Lasso no `solver`.

## 2. The multi-class exponential loss

Encode class $c$ as the **symmetric** code vector $\mathbf{y} \in \mathbb{R}^K$:

$$y_k = \begin{cases} 1 & k = c \\[1mm] -\dfrac{1}{K-1} & k \ne c \end{cases}
  \qquad\Longrightarrow\qquad \sum_{k} y_k = 0 .$$

The ensemble is an additive model
$\mathbf{f}(x) = \sum_{m=1}^{M} \beta_m\, \mathbf{g}_m(x)$, where each
$\mathbf{g}_m(x)$ is the code vector of $T_m(x)$, subject to the
sum-to-zero constraint $\sum_k f_k(x) = 0$. The loss is

$$L(\mathbf{y}, \mathbf{f}) = \exp\!\Big(-\tfrac{1}{K}\,\mathbf{y}^\top \mathbf{f}\Big).$$

For $K = 2$ (codes $\pm 1$) this is exactly $e^{-y f}$, the loss of
classic AdaBoost. Minimising the *population* exponential loss recovers a
scaled log-odds — the multi-class analogue of AdaBoost's Bayes
consistency (Zhu et al., §3) — so it is a sensible surrogate for the 0–1
loss.

## 3. Forward stagewise minimisation

At round $m$, given $\mathbf{f}^{(m-1)}$, solve

$$(\beta_m, \mathbf{g}_m)
  = \arg\min_{\beta,\, \mathbf{g}}
    \sum_{i=1}^{n} \exp\!\Big(-\tfrac{1}{K}\,\mathbf{y}_i^\top
      \big(\mathbf{f}^{(m-1)}(x_i) + \beta\,\mathbf{g}(x_i)\big)\Big).$$

Define the **sample weight**
$w_i = \exp\!\big(-\tfrac{1}{K}\mathbf{y}_i^\top \mathbf{f}^{(m-1)}(x_i)\big)$,
so the objective is
$\sum_i w_i \exp\!\big(-\tfrac{\beta}{K}\mathbf{y}_i^\top \mathbf{g}(x_i)\big)$.
For the symmetric codes, $\mathbf{y}_i^\top \mathbf{g}(x_i)$ takes just two
values:

$$\mathbf{y}_i^\top \mathbf{g}(x_i)
  = \begin{cases}
      \dfrac{K}{K-1} & T_m(x_i) = c_i \quad\text{(correct)}\\[3mm]
      -\dfrac{K}{(K-1)^2} & T_m(x_i) \ne c_i \quad\text{(wrong)}
    \end{cases}$$

*(check, wrong case: the dot product is
$-\tfrac{2}{K-1} + \tfrac{K-2}{(K-1)^2} = \tfrac{-2(K-1) + (K-2)}{(K-1)^2}
= -\tfrac{K}{(K-1)^2}$.)*

With weights normalised ($\sum_i w_i = 1$) and the **weighted error**

$$\mathrm{err} = \sum_{i:\, T_m(x_i) \ne c_i} w_i ,$$

the objective as a function of $\beta$ is

$$\Phi(\beta)
  = (1 - \mathrm{err})\, e^{-\beta/(K-1)}
  + \mathrm{err}\, e^{\beta/(K-1)^2}.$$

**The weak learner $\mathbf{g}_m$.** $\Phi$ is increasing in $\mathrm{err}$
for any $\beta > 0$, so $\mathbf{g}_m$ is the classifier that **minimises
the weighted error** — i.e. fit $T_m$ on $(X, y)$ with
`sample_weight` $= w$. This is exactly the hook added in
[`decision_tree.md`](decision_tree.md) §3b.

**The step size $\beta_m$.** Set $\Phi'(\beta) = 0$:

$$\frac{\mathrm{err}}{(K-1)^2}\, e^{\beta/(K-1)^2}
  = \frac{1 - \mathrm{err}}{K-1}\, e^{-\beta/(K-1)}
  \;\Longrightarrow\;
  \beta \cdot \frac{K}{(K-1)^2}
  = \log\frac{1 - \mathrm{err}}{\mathrm{err}} + \log(K-1),$$

so

$$\beta_m = \frac{(K-1)^2}{K}
  \left(\log\frac{1 - \mathrm{err}_m}{\mathrm{err}_m} + \log(K-1)\right).$$

## 4. The weight update and the practical $\alpha_m$

$w_i^{(m+1)} = w_i^{(m)}
  \exp\!\big(-\tfrac{\beta_m}{K}\,\mathbf{y}_i^\top \mathbf{g}_m(x_i)\big)$.
The exponent is $-\tfrac{\beta_m}{K-1}$ (correct) or
$+\tfrac{\beta_m}{(K-1)^2}$ (wrong). Adding the constant
$\tfrac{\beta_m}{K-1}$ to both — a global rescale that cancels in the
renormalisation — sends "correct" to $0$ and "wrong" to
$\tfrac{\beta_m K}{(K-1)^2}$. Define

$$\boxed{\;\alpha_m
  = \frac{\beta_m K}{(K-1)^2}
  = \log\frac{1 - \mathrm{err}_m}{\mathrm{err}_m} + \log(K-1)\;}$$

and the update is

$$w_i \;\leftarrow\; w_i \,
  \exp\!\big(\eta\,\alpha_m\,\mathbb{1}[\,c_i \ne T_m(x_i)\,]\big),
  \qquad\text{then renormalise to } \textstyle\sum_i w_i = 1 .$$

$\eta = $ `learning_rate` shrinks each step. Misclassified rows have their
weight multiplied by $e^{\eta\alpha_m} > 1$, so round $m+1$ concentrates on
what round $m$ got wrong — the entire idea of boosting.

## 5. Prediction

The trained model classifies $\arg\max_k f_k(x)$. Expanding
$f_k(x) = \sum_m \beta_m\big(\tfrac{K}{K-1}\mathbb{1}[T_m(x)=k]
- \tfrac{1}{K-1}\big)$; the $-\tfrac{1}{K-1}\sum_m\beta_m$ term is constant
across $k$, and $\beta_m \propto \alpha_m$ with a positive constant, so

$$\hat{y}(x) = \mathcal{C}\Big[\arg\max_k
  \sum_{m=1}^{M} \alpha_m\, \mathbb{1}[\,T_m(x) = k\,]\Big].$$

Argmax ties resolve to the lowest class label (`classes_` sorted), as
everywhere in the codebase.

- **`decision_function(X)`** returns the score
  $\tilde f_k(x) = \dfrac{1}{\sum_m \alpha_m}
   \sum_m \alpha_m\Big(\mathbb{1}[T_m(x)=k]
   - \tfrac{1}{K-1}\big(1 - \mathbb{1}[T_m(x)=k]\big)\Big)$
  — i.e. each stump adds $+\alpha_m$ to the column of the class it predicts
  and $-\alpha_m/(K-1)$ to every other column, so its contribution sums to
  zero across classes (this is scikit-learn's exact form; it is
  $\tfrac{K}{K-1}$ times the $\sum_m \alpha_m(\mathbb{1}[T_m=k]-\tfrac1K)$
  of §2, and $\arg\max_k$ is unchanged by that scale). Shape $(n, K)$,
  collapsed to $\tilde f_1 - \tilde f_0$ (shape $(n,)$) for $K = 2$,
  matching scikit-learn's sign convention.
- **`predict_proba(X)`** is $\mathrm{softmax}\big(\tilde f(x) / (K-1)\big)$
  — scikit-learn's monotone transform (for $K = 2$ this collapses to
  $\sigma(\tilde f_1 - \tilde f_0)$). A **score calibration, not a
  posterior**: monotone in the vote margin and nothing more; the docstring
  says so.
- **`staged_predict(X)` / `staged_score(X, y)`** yield the prediction /
  accuracy of the partial ensemble $T_1, \dots, T_t$ for
  $t = 1, \dots, M$ — the canonical way to see the boosting curve.

## 6. $K = 2$ reduces to classic discrete AdaBoost

With $K = 2$, $\log(K-1) = 0$, so
$\alpha_m = \log\frac{1 - \mathrm{err}_m}{\mathrm{err}_m}$ and error weights
are multiplied by $e^{\alpha_m} = \frac{1 - \mathrm{err}_m}{\mathrm{err}_m}$.
Classic discrete AdaBoost (Freund–Schapire) uses
$\alpha_m^{\text{FS}} = \tfrac{1}{2}\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$
with labels $\pm 1$ and update $w_i \leftarrow w_i e^{-\alpha^{\text{FS}} y_i h_m(x_i)}$,
i.e. multiply by $e^{2\alpha^{\text{FS}}} = \frac{1-\mathrm{err}_m}{\mathrm{err}_m}$
on errors. **Identical weight dynamics, identical predictions**; the SAMME
$\alpha$ is exactly twice the Freund–Schapire one, the factor absorbed by
the symmetric encoding. This equivalence is a test (§10).

## 7. Convergence and degenerate rounds

**Why it works.** Each round drives the training exponential loss down as
long as the weak learner beats random guessing, $\mathrm{err}_m < 1 - 1/K$
(so $\alpha_m > 0$); the training 0–1 error is upper-bounded by the
exponential loss and follows it down (Zhu et al., Thm 1). Boosting *can*
eventually overfit — unlike a random forest — because it keeps driving the
margin, so `n_estimators` is a real hyperparameter, not a
"more-is-always-safe" knob.

**Degenerate rounds** — matched to scikit-learn's SAMME exactly:

- **$\mathrm{err}_m = 0$** (the weak learner fits the reweighted data
  perfectly): record $\alpha_m = 1$, $\mathrm{err}_m = 0$, keep the
  estimator, and **stop** — no further rounds can improve on a perfect
  fit, and $\alpha_m \to \infty$ is undefined.
- **$\mathrm{err}_m \ge 1 - 1/K$** (no better than random): **discard**
  that estimator. If it was the first round, **raise** `ValueError`
  ("base estimator worse than random"); otherwise **stop** with the
  estimators collected so far.
- The sample weights are renormalised each round; if their sum is
  non-positive, stop.

`estimators_`, `estimator_weights_`, `estimator_errors_` are trimmed to the
number of rounds actually run.

## 8. What it costs, and omissions

- **Fitting** is $M$ sequential tree builds (each round depends on the
  previous round's weights) — no parallelism, by nature. A stump build is
  $O(d\, n \log n)$, so the default is cheap.
- **Serial only** — there is nothing to parallelise across rounds; the
  base tree builds are already vectorised.
- **`AdaBoostRegressor`** (AdaBoost.R2 — a different algorithm) is
  deferred, like `RandomForestRegressor` / `KNeighborsRegressor`.
- **`SAMME.R`** (the real-valued variant using `predict_proba`) is not
  implemented — removed from scikit-learn 1.6; SAMME is the method.
- **A general `estimator=` parameter** (+ `clone`) is omitted: the base is
  always a `DecisionTreeClassifier`, with `max_depth` exposed.
- **`sample_weight` sparsity**, **`min_weight_fraction_leaf`** on the base
  — not carried through.

## 9. Algorithm (pseudocode)

```
AdaBoostClassifier(n_estimators=50, learning_rate=1.0, max_depth=1)

_weighted_error(y_true, y_pred, w):        # w already sums to 1
    return sum(w[y_true != y_pred])

_samme_alpha(err, K, learning_rate):
    return learning_rate * (log((1 - err) / err) + log(K - 1))

_samme_decision(estimators, weights, classes, X):   # -> (m, K), zero-sum rows, /Σ weights
    K = len(classes)
    score = zeros((len(X), K))
    for tree, a in zip(estimators, weights):
        pred_idx = searchsorted(classes, tree.predict(X))
        contribution = full((len(X), K), -1 / (K - 1))   # -a/(K-1) off the predicted class
        contribution[arange(len(X)), pred_idx] = 1        # +a on the predicted class
        score += a * contribution
    return score / weights.sum()

fit(X, y, sample_weight=None):
    validate: n_estimators >= 1;  learning_rate > 0;  max_depth >= 1
    X, y = check_X_y(X, y)
    w = check_sample_weight(sample_weight, n) / sum(...)     # normalise to 1
    classes_ = unique(y);  n_classes_ = K = len(classes_)

    estimators_, estimator_weights_, estimator_errors_ = [], [], []
    for m in range(n_estimators):
        tree = DecisionTreeClassifier(max_depth=max_depth).fit(X, y, sample_weight=w)
        err  = _weighted_error(y, tree.predict(X), w)

        if err <= 0:                                  # perfect on reweighted data
            estimators_.append(tree); estimator_weights_.append(1.0)
            estimator_errors_.append(0.0);  break
        if err >= 1 - 1 / K:                          # worse than random
            if len(estimators_) == 0:
                raise ValueError("base estimator worse than random")
            break

        alpha = _samme_alpha(err, K, learning_rate)
        w = w * exp(alpha * (tree.predict(X) != y))
        w = w / w.sum()                               # renormalise
        estimators_.append(tree)
        estimator_weights_.append(alpha)
        estimator_errors_.append(err)

    estimator_weights_ = array(estimator_weights_)
    estimator_errors_  = array(estimator_errors_)
    feature_importances_ = normalise(
        Σ_m (estimator_weights_[m] / Σ estimator_weights_) · estimators_[m].feature_importances_)
    return self

decision_function(X):  d = _samme_decision(estimators_, estimator_weights_, classes_, X)
                       return d[:, 1] - d[:, 0] if K == 2 else d
predict(X):            classes_[argmax(_samme_decision(...), axis=1)]
predict_proba(X):      softmax(_samme_decision(...) / (K - 1), axis=1)
staged_predict(X):     for t in 1..M: yield classes_[argmax(_samme_decision(estimators_[:t], ...), 1)]
score(X, y):           accuracy_score(y, predict(X))
```

Module-level `_weighted_error`, `_samme_alpha`, `_samme_decision` are
factored out and unit-tested directly, mirroring the `_*` helpers in
`linear/`, `neighbors/`, `naive_bayes/`, `tree/`, and `ensemble/`.

## 10. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | `_samme_alpha` equals $\log\frac{1-e}{e} + \log(K-1)$ on hand values; $K = 2$ drops the $\log(K-1)$ term; `learning_rate` scales it linearly. `_weighted_error` on a hand-weighted mini-example. |
| Derivation stand-in (no differentiable objective) | after one round, the new weights equal an independent recomputation of $w\,e^{\eta\alpha\,\mathbb{1}[\text{miss}]}$ renormalised; a $K = 2$ run reproduces a from-scratch classic discrete-AdaBoost (weights $2\times$, identical predictions). |
| Reduces to the base | `n_estimators=1` → predictions identical to a lone `DecisionTreeClassifier(max_depth=1)` on the same (unweighted) data. |
| Boosting behaviour | on `make_moons(noise=0.3)`: staged training accuracy is non-decreasing within noise; the final ensemble beats a single stump on held-out data by a clear margin; `staged_score` is monotone-ish upward early. |
| Early stop — perfect | data a single stump separates → `len(estimators_) == 1`, `estimator_errors_[0] == 0`. |
| Early stop — worse than random | a contrived set where the stump cannot beat $1 - 1/K$ → `ValueError` on the first round; if it only degrades later, the ensemble stops early and still predicts. |
| `decision_function` / `predict_proba` | shapes ($(n,)$ binary, $(n, K)$ multiclass); `predict == argmax` of the score; `predict_proba` rows sum to 1 and lie in $[0, 1]$; the binary `decision_function` sign agrees with `predict`. |
| `feature_importances_` | shape $(d,)$, sums to 1, zero for a feature no stump splits on; equals the $\alpha$-weighted mean of the per-stump importances. |
| `staged_predict` | yields `len(estimators_)` arrays; the last equals `predict`. |
| Determinism | same data → identical fit (no RNG on the default path). |
| Contract | `NotFittedError` before `fit`; `fit` returns self; hyperparameters unmutated; feature-count mismatch in `predict` raises; `n_estimators < 1`, `learning_rate <= 0`, `max_depth < 1`, bad `sample_weight` all raise; `repr` round-trips. |
| Reference (`-m reference`) | vs `sklearn.ensemble.AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm="SAMME", ...)`, binary and 3-class: `estimator_errors_` and `estimator_weights_` within `atol` on the first few rounds; held-out predictions agree on $\ge 97\%$ of rows (a stump equal-gain tie can send the two down different paths); `predict_proba` close in $\ell_\infty$. |

Plus `examples/adaboost.py` — seeded, runnable: the staged test-accuracy
curve, single stump vs. the boosted ensemble, and `estimator_weights_` /
`estimator_errors_` over the rounds.

## Scope — what this estimator omits (and why)

- **`AdaBoostRegressor`** (AdaBoost.R2) — a separate algorithm, deferred.
- **`SAMME.R`** — removed from scikit-learn 1.6; SAMME is the method.
- **general `estimator=` + `clone`** — base is always a
  `DecisionTreeClassifier` stump; `max_depth` is the one exposed knob.
- **`n_jobs`** — boosting rounds are inherently sequential.

## References

- Zhu, Zou, Rosset & Hastie (2009), "Multi-class AdaBoost", *Statistics
  and its Interface* 2(3) — the SAMME derivation.
- Freund & Schapire (1997), "A decision-theoretic generalization of
  on-line learning and an application to boosting", *JCSS* 55(1).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §10 (Boosting and Additive Trees).
- scikit-learn User Guide, §1.11.3 (AdaBoost).
