# Linear SVM (soft-margin, primal, subgradient descent)

The finalized derivation walkthrough for `scratchgrad.svm.LinearSVM`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md); the intercept treatment mirrors
[`ridge.md`](ridge.md)/[`logistic_regression.md`](logistic_regression.md) and
the non-differentiable-objective handling mirrors
[`lasso.md`](lasso.md)'s L1 treatment.

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — $n$ samples, $d$ features |
| $y$ | $(n,)$ | binary labels, re-encoded internally to $\{-1, +1\}$ |
| $w$ | $(d,)$ | weight vector (`coef_`) |
| $b$ | scalar | intercept (`intercept_`) |
| $z_i = w^\top x_i + b$ | scalar | decision score for sample $i$ |
| $m_i = y_i z_i$ | scalar | **margin** of sample $i$ — positive iff correctly classified, $\ge 1$ iff outside the margin band |
| $C$ | scalar $> 0$ | regularisation strength (`C`) |
| $\eta_t$ | scalar | step size at iteration $t$ |

**Model.** $f(x) = w^\top x + b$; $\hat{y} = \mathrm{sign}(f(x))$, mapped back
to the two original labels via `classes_`.

**Binary only.** As with `LogisticRegression` (`plan.md` M1), `fit` requires
`y` to hold exactly two distinct values; the smaller sorts to $-1$, the
larger to $+1$. One-vs-rest multiclass is deferred — not needed until a
concrete downstream use calls for it.

The labels are re-encoded to $\{-1, +1\}$ rather than $\{0, 1\}$
(`LogisticRegression`'s convention) because the hinge loss (§2) is stated
most naturally in terms of the signed margin $y_i z_i$ — that is standard
for every SVM reference (Cortes & Vapnik 1995; Hastie/Tibshirani/Friedman
§12.2; Bishop §7.1).

## 2. Objective — soft-margin primal

$$\boxed{\;J(w, b) = \frac{1}{2}\lVert w\rVert_2^2 \;+\; C\sum_{i=1}^n
  \max\!\big(0,\; 1 - y_i(w^\top x_i + b)\big)\;}$$

This is exactly `sklearn.svm.LinearSVC(loss="hinge")`'s documented primal
objective — `C` multiplies the **raw sum** of hinge losses (no $\tfrac1n$),
and only $w$ is regularised, never $b$ (same intercept convention as
`Ridge`/`LogisticRegression`, and the same reason: penalising $b$ would make
the fit depend on the arbitrary origin of the feature space). So our `C`
equals scikit-learn's `LinearSVC`'s `C` directly, with no rescaling — the
same "match this estimator's own scikit-learn counterpart's convention"
policy `ridge.md`/`logistic_regression.md` state, applied here.

**Geometric reading.** Minimising $\lVert w\rVert$ is maximising the margin
width $2/\lVert w\rVert$ between the two classes' supporting hyperplanes
$w^\top x + b = \pm 1$. The hinge term $\max(0, 1 - y_i z_i)$ is zero for any
point correctly classified with margin $\ge 1$, and grows linearly for every
point that is either misclassified or sits inside the margin band. $C$
controls the trade-off: large $C$ penalises margin violations heavily (a
"harder" margin, more sensitive to individual points), small $C$ tolerates
more violations in exchange for a wider margin.

**Convexity.** $\tfrac12\lVert w\rVert^2$ is strictly convex in $w$;
$\max(0, 1 - y_i z_i)$ is a max of two affine functions of $(w, b)$, hence
convex; a sum of convex functions is convex. So $J$ is convex (strictly so
in $w$), and any subgradient stationary point — $0 \in \partial J(w, b)$ —
is a global minimum.

## 3. Non-differentiability → subgradient

$\max(0, 1 - y_i z_i)$ has a kink exactly where $y_i z_i = 1$ — a sample
sitting precisely on the margin boundary. Its subdifferential with respect
to $w$ (chain rule through $z_i = w^\top x_i + b$, $\partial_w z_i = x_i$):

$$\partial_w \max(0, 1 - y_i z_i) = \begin{cases}
  -y_i x_i & y_i z_i < 1 \quad \text{(violated: inside the margin or misclassified)} \\
  0 & y_i z_i > 1 \quad \text{(satisfied: strictly outside the margin)} \\
  [-y_i x_i,\ 0] & y_i z_i = 1 \quad \text{(exactly on the margin — a whole interval is valid)}
\end{cases}$$

and symmetrically for $\partial_b$, since $\partial_b z_i = 1$. Picking the
flat-side value ($0$) at the exact kink — the same "pick one consistent
subgradient" move `lasso.md` §4 makes at $w_j = 0$ — gives an implementable
subgradient of the full objective. Let
$\mathcal{V} = \{i : y_i z_i < 1\}$ (the violated set):

$$\boxed{\;g_w = w - C\sum_{i \in \mathcal{V}} y_i x_i, \qquad
  g_b = -C\sum_{i \in \mathcal{V}} y_i\;}$$

This is the vector-form analogue of Lasso's KKT condition: no closed form
for the whole $(w, b)$ at once (unlike Ridge's normal equation), so an
iterative method is required.

## 4. Why the subgradient method needs a diminishing step size and best-iterate tracking

Ordinary gradient descent's guarantee that $J$ decreases every step relies
on **smoothness** (an L-Lipschitz gradient) — this does not hold at a kink.
With a *fixed* step size, subgradient descent on a non-smooth convex
function can permanently oscillate around the optimum instead of
converging to it (Boyd & Vandenberghe's convex optimisation notes; Boyd,
Xiao & Mutapcic, *Subgradient Methods*, notes for EE364b). Two standard
fixes, both used here, together give a real convergence guarantee:

1. **Diminishing step size** $\eta_t = \mathrm{lr}/\sqrt{t}$ for $t = 1, 2,
   \dots$. This satisfies $\sum_t \eta_t = \infty$ (the iterate can still
   travel arbitrarily far in total, so it is not stuck too early) and
   $\eta_t \to 0$ (the steps shrink, damping the oscillation at a kink).
   Under these conditions and a bounded-subgradient assumption, the
   subgradient method's **best iterate so far** converges to the true
   minimum of a convex $J$.
2. **Track the best $(w, b)$ seen**, not just the final one — because $J$
   is not guaranteed to *decrease* at every single step (unlike smooth
   gradient descent), only to trend toward the optimum in the limit. A
   run that happens to land on a locally worse iterate at `max_iter` would
   silently regress if `fit` just returned the last step; keeping the
   minimum-$J$ iterate (a "pocket" update) is the standard practical fix
   and is what makes the method usable at a *finite* `max_iter`.

**A consequence worth noting**: unlike `LinearRegression`/`Ridge`/
`LogisticRegression`, this project's `LinearSVM` has **no `random_state`
parameter at all** — $w^{(0)} = 0$, $b^{(0)} = 0$ deterministically, every
step is a deterministic function of the data, and there is no
initialization or sampling randomness anywhere in the algorithm. `fit` is
therefore byte-identical for the same `(X, y, hyperparameters)` on every
run, the same way `DBSCAN` and `PCA` are (`plan.md`/`handoff.md` note this
as a rare property among this project's estimators).

## 5. Stopping — fixed `max_iter`, no `tol`

Unlike the tol-based early stopping in `LinearRegression`/`Ridge`/`Lasso`/
`LogisticRegression`, this estimator runs a **fixed** number of subgradient
steps, `max_iter`, with no convergence tolerance. Two reasons:

- The natural stopping signal for smooth solvers — "the gradient norm is
  near zero" — does not transfer cleanly here: a subgradient of $0$ genuinely
  can occur away from the optimum's *unique* subgradient value (the
  subdifferential at a kink is a whole set, and which member the descent
  picks at a given step is not itself a distance-to-optimum signal the way
  a smooth gradient's norm is).
- This matches how subgradient/Pegasos-style methods (Shalev-Shwartz,
  Singer, Srebro & Cotter, 2011, *Pegasos: primal estimated sub-gradient
  solver for SVM*) are conventionally run in practice: a fixed epoch budget,
  not a convergence test.

So `n_iter_` is simply `max_iter` after every `fit` call — it records how
many subgradient steps ran, not a claim of convergence detection. `coef_`/
`intercept_` are the **best-$J$** iterate found across those steps (§4),
not necessarily the last one.

## 6. Predict, decision function, and score

- `decision_function(X)` → $z = Xw + b$, shape $(n,)$. Its sign is the
  predicted class; its magnitude is (unnormalised) confidence.
- `predict(X)` → `classes_[(decision_function(X) >= 0).astype(int)]` — sign
  of the score maps back to the two original labels.
- `score(X, y)` → `accuracy_score` (`scratchgrad.metrics`).
- **No `predict_proba`.** A hinge-loss SVM has no native probabilistic
  output — scikit-learn's own `LinearSVC` doesn't expose one either without
  wrapping it in a separate Platt-scaling calibrator
  (`sklearn.calibration.CalibratedClassifierCV`), which fits an auxiliary
  logistic regression on the decision scores. That is new math bolted onto
  a different model, not part of the SVM itself, so it is out of scope here.

## 7. scikit-learn parity strategy

`sklearn.svm.LinearSVC` solves the **dual** problem via liblinear's
coordinate descent — a fundamentally different algorithm from this
project's primal subgradient descent, the same situation
`gradient_boosting.md` describes for `friedman_mse` vs. `squared_error`
trees. So there is **no exact parity path** here (unlike `DBSCAN`/`PCA`,
which have one because they have zero algorithmic choices beyond the
definition itself). Reference tests compare outcomes — held-out accuracy
and decision-boundary agreement — within a tolerance, the same tier
`RandomForest`/`GradientBoosting` use, not `coef_`/`intercept_` equality.

## 8. Algorithm (pseudocode)

```
LinearSVM(C=1.0, fit_intercept=True, lr=0.01, max_iter=1000)

fit(X, y):
    if C <= 0:        raise ValueError
    if lr <= 0:        raise ValueError
    if max_iter < 1:   raise ValueError
    X, y = check_X_y(X, y)
    classes_ = unique(y);  if len(classes_) != 2: raise ValueError
    t = where(y == classes_[1], 1.0, -1.0)          # labels -> {-1, +1}

    w, b = zeros(d), 0.0
    w_best, b_best = w, b
    J_best = _svm_objective(X, t, w, b, C)           # = C * n at w=b=0

    for step in 1 .. max_iter:
        g_w, g_b = _svm_subgradient(X, t, w, b, C)
        eta = lr / sqrt(step)
        w = w - eta * g_w
        if fit_intercept: b = b - eta * g_b
        J = _svm_objective(X, t, w, b, C)
        if J < J_best:
            J_best, w_best, b_best = J, w, b

    coef_      = w_best
    intercept_ = b_best if fit_intercept else 0.0
    n_iter_    = max_iter
    return self

_svm_objective(X, y, w, b, C):
    margin = y * (X @ w + b)                          # m_i = y_i z_i
    return 0.5 * (w @ w) + C * sum(maximum(0, 1 - margin))

_svm_subgradient(X, y, w, b, C):
    margin = y * (X @ w + b)
    violated = margin < 1                             # V = {i : m_i < 1}
    g_w = w - C * X[violated].T @ y[violated]
    g_b = -C * sum(y[violated])
    return g_w, g_b

decision_function(X): return X @ coef_ + intercept_
predict(X):           return classes_[(decision_function(X) >= 0).astype(int)]
score(X, y):           return accuracy_score(y, predict(X))
```

Module-level `_svm_objective(X, y, w, b, C)` and `_svm_subgradient(X, y, w,
b, C)` factor out the math so tests can target them directly, mirroring
`_ridge_objective`/`_ridge_gradient`. Note `X[violated]` on an empty
`violated` mask is simply an empty `(0, d)` slice, so
`X[violated].T @ y[violated]` is `zeros(d)` with no special-case branch
needed — the vectorised form already handles "nothing violated" correctly.

## 9. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | At $w=0, b=0$: every margin is $0 < 1$, so all $n$ points are violated and $J = Cn$, $g_w = -C\sum_i y_i x_i$, $g_b = -C\sum_i y_i$ — checked against a hand computation. |
| Gradient | At a random $(w, b)$ with no sample exactly on the margin (probability-zero event for continuous data, so a fixed seed is safe), `_svm_subgradient` matches central finite differences of `_svm_objective` (`tests/helpers/gradcheck.py`) — valid because the hinge is smooth in a neighborhood of any non-kink point, so the subgradient there *is* the ordinary gradient. |
| Convexity | `_svm_objective` is convex: for random $(w_1, b_1)$, $(w_2, b_2)$ and $\lambda \in (0,1)$, $J(\lambda\theta_1 + (1-\lambda)\theta_2) \le \lambda J(\theta_1) + (1-\lambda) J(\theta_2)$. |
| Best-iterate | The best-$J$ tracked across steps is non-increasing by construction; a stress case with a large `lr` (which makes the *last* iterate's $J$ worse than an earlier one) confirms `coef_`/`intercept_` come from the best step, not the final one. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; `coef_` has shape `(d,)`; feature-count mismatch in `predict` raises; non-binary `y` raises; `C <= 0` raises; `lr <= 0` raises; `max_iter < 1` raises; `score` is accuracy; `predict_proba` does not exist; repr round-trips. |
| Determinism | Two `fit` calls on identical data produce byte-identical `coef_`/`intercept_` (no `random_state` needed — see §4). |
| Behavioral | On `make_blobs(centers=2)` (near-separable) training accuracy $> 0.95$; on noisy `make_moons` accuracy beats the majority-class baseline by a clear margin; a larger `C` does not increase $\lVert w\rVert_2$ by orders of magnitude less than a smaller `C` on the same separable data (margin hardens as `C` grows). |
| Edge | `fit_intercept=False`; single feature; a perfectly separable toy dataset recovers the correct sign of `coef_`. |
| Reference (`-m reference`) | Held-out accuracy within a tolerance of `sklearn.svm.LinearSVC(loss="hinge")`; decision-boundary sign agreement on held-out points within a tolerance — no `coef_` comparison (different solver, §7). |

Plus `examples/linear_svm.py` — seeded blobs: fit, report the margin
($1/\lVert w\rVert_2$), accuracy, and a `C` sweep showing the margin/
violation trade-off.

## References

- Cortes, C. & Vapnik, V. (1995), "Support-Vector Networks", *Machine
  Learning* 20(3) — the original soft-margin SVM formulation.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §12.2 (the support vector classifier, hinge loss).
- Bishop, *Pattern Recognition and Machine Learning*, §7.1 (maximum
  margin classifiers).
- Boyd, S., Xiao, L. & Mutapcic, A., *Subgradient Methods*, Stanford
  EE364b lecture notes — the diminishing-step-size convergence argument
  used in §4.
- Shalev-Shwartz, S., Singer, Y., Srebro, N. & Cotter, A. (2011),
  "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", *Mathematical
  Programming* 127(1) — fixed-epoch subgradient training in practice.
