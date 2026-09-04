# Lasso regression (L1-penalised least squares, coordinate descent)

The finalized derivation walkthrough for `scratchgrad.linear.Lasso`, written
per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and mirrors
[`linear_regression.md`](linear_regression.md) and [`ridge.md`](ridge.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — $n$ samples, $d$ features |
| $y$ | $(n,)$ | regression targets |
| $w$ | $(d,)$ | weight vector (`coef_`) |
| $b$ | scalar | intercept (`intercept_`) |
| $x_j$ | $(n,)$ | column $j$ of $X$ |
| $\alpha$ | scalar $\ge 0$ | regularisation strength (`alpha`) |
| $r$ | $(n,)$ | residual $y - Xw$ (on centered data) |
| $r_j$ | $(n,)$ | partial residual — $r$ with feature $j$'s term added back |

The model is unchanged from OLS: $\hat{y} = Xw + b$. When
`fit_intercept=False` the data is not centered and `intercept_` is fixed at
$0$. The hyperparameter is named **`alpha`** (`lambda` is a reserved word),
matching `sklearn.linear_model.Lasso(alpha=...)`.

## 2. Objective

$$J(w, b) = \frac{1}{2n}\lVert y - Xw - b\mathbf{1}\rVert_2^2
          \;+\; \alpha\lVert w\rVert_1$$

Three points:

**The L1 penalty is on $w$ only, never the intercept $b$** — as in ridge,
penalising $b$ would make the fit depend on the target's origin.

**The $\tfrac{1}{2n}$ prefactor matches scikit-learn's `Lasso`.** Its
documented objective is $\frac{1}{2\,n_\text{samples}}\lVert y - Xw\rVert_2^2
+ \alpha\lVert w\rVert_1$, so our `alpha` equals theirs directly. This is a
*different* convention than this project's `Ridge` (which follows
scikit-learn's `Ridge`: $\lVert y - Xw\rVert_2^2 + \alpha\lVert w\rVert_2^2$,
no $\tfrac{1}{2n}$). Each estimator matches its own scikit-learn
counterpart's `alpha`; that is the priority. The $\tfrac12$ also cancels the
factor of 2 from differentiating the square, so the coordinate update in §4
comes out clean.

**$J$ is convex but not differentiable.** The quadratic term is convex and
smooth; $\lVert w\rVert_1 = \sum_j \lvert w_j\rvert$ is convex but has a kink
at every $w_j = 0$. So there is no gradient to set to zero, no normal
equation, and no closed-form solution for the whole vector at once.

## 3. Why L1 gives sparsity (and L2 does not)

Constrained form: minimise $\lVert y - Xw\rVert_2^2$ subject to
$\lVert w\rVert_1 \le t$. The L1 ball is a cross-polytope — a diamond in 2-D,
with **vertices on the coordinate axes**. The elliptical level sets of the
least-squares term generically first touch that ball at a **vertex**, where
some coordinates are exactly $0$. Ridge's L2 ball is round: it has no
vertices, the contact point almost never lands exactly on an axis, and
coefficients shrink toward $0$ without reaching it.

Algebraically (§4) this is the soft-thresholding operator: it maps an entire
interval $[-\alpha, \alpha]$ of inputs to exactly $0$. Larger $\alpha$ ⇒ more
zeros ⇒ Lasso performs variable selection.

## 4. Coordinate descent and the soft-thresholding operator

**Why coordinate descent works here.** $J$ is convex, and its non-smooth
part $\alpha\sum_j \lvert w_j\rvert$ is *separable* — a sum of one-variable
functions. For a convex function that is smooth plus separable-non-smooth,
cyclically minimising over one coordinate at a time converges to the global
minimum. And each one-coordinate subproblem has a closed form.

**One coordinate.** Work on centered data (intercept handled in §5), so drop
$b$. Fix every $w_k$ for $k \ne j$ and define the **partial residual**

$$r_j = y - \sum_{k \ne j} w_k x_k$$

As a function of $w_j$ alone,

$$f(w_j) = \frac{1}{2n}\lVert r_j - w_j x_j\rVert_2^2
         + \alpha\lvert w_j\rvert + \text{const}$$

Expand the quadratic and differentiate its (smooth) part:

$$\frac{\mathrm{d}}{\mathrm{d}w_j}\left[
  \frac{1}{2n}\big(\lVert r_j\rVert^2 - 2 w_j\, x_j^\top r_j
  + w_j^2\lVert x_j\rVert^2\big)\right]
  = -\rho_j + z_j w_j$$

with

$$\rho_j = \frac{1}{n}\, x_j^\top r_j, \qquad
  z_j = \frac{1}{n}\lVert x_j\rVert_2^2$$

The subdifferential of $\alpha\lvert w_j\rvert$ is $\alpha\operatorname{sign}
(w_j)$ for $w_j \ne 0$ and the interval $[-\alpha, \alpha]$ at $w_j = 0$. The
optimality condition $0 \in \partial f(w_j)$ is

$$0 \in z_j w_j - \rho_j + \alpha\,\partial\lvert w_j\rvert$$

Case-split:

| condition | consistent solution |
| --- | --- |
| $\rho_j > \alpha$ | $w_j = (\rho_j - \alpha) / z_j > 0$ |
| $\rho_j < -\alpha$ | $w_j = (\rho_j + \alpha) / z_j < 0$ |
| $\lvert\rho_j\rvert \le \alpha$ | $w_j = 0$ (then $\rho_j \in \alpha[-1,1]$ holds) |

which is exactly

$$\boxed{\;w_j \;=\; \frac{S(\rho_j,\ \alpha)}{z_j}, \qquad
  S(\rho, \alpha) = \operatorname{sign}(\rho)\,\big(\lvert\rho\rvert
  - \alpha\big)_+\;}$$

$S$ is the **soft-thresholding operator** — shrink $\rho$ toward $0$ by
$\alpha$, and snap to exactly $0$ inside $[-\alpha, \alpha]$. It is the
minimiser of $\frac12(w - \rho)^2 + \alpha\lvert w\rvert$.

**Residual maintenance.** Recomputing each partial residual $r_j$ from
scratch is $O(nd)$ per coordinate, $O(nd^2)$ per sweep. Instead keep the
full residual $r = y - Xw$ current. Since $r_j = r + w_j x_j$,

$$\rho_j = \frac{1}{n} x_j^\top r + z_j w_j^{\text{old}}$$

compute $w_j^{\text{new}}$, then update $r \mathrel{-}= (w_j^{\text{new}}
- w_j^{\text{old}})\,x_j$. $z_j$ is constant across sweeps — precompute it
once. A sweep is now $O(nd)$. This is the standard "coordinate descent for
Lasso" (glmnet, scikit-learn).

## 5. Intercept

Same argument as ridge: for any fixed $w$, the unpenalised $b$ that
minimises $J$ is the residual mean, and substituting it back gives the same
objective on centered data. So when `fit_intercept=True`:

1. $\bar x = X.\text{mean}(0)$, $\bar y = y.\text{mean}()$; center
   $X_c = X - \bar x$, $y_c = y - \bar y$.
2. Run coordinate descent on $(X_c, y_c)$ with no intercept → $w$.
3. $b = \bar y - \bar x^\top w$.

Identical to scikit-learn's `Lasso`. `fit_intercept=False` skips centering
and sets $b = 0$.

`alpha` penalises every $w_j$ equally, so on raw, differently-scaled features
it regularises them unevenly. This estimator does not rescale inputs inside
`fit` (`plan.md` §1) — standardise with `StandardScaler` first if you want
`alpha` to be comparable across features. A constant feature has
$z_j = 0$ after centering; its weight is left at $0$ and the coordinate is
skipped.

## 6. `alpha = 0` reduces to OLS

At $\alpha = 0$, $S(\rho, 0) = \rho$ and the update is plain least-squares
coordinate descent, which converges to the OLS solution (unique when $X_c$
has full column rank). `Lasso(alpha=0)` therefore matches
`LinearRegression` given enough sweeps. Degenerate on collinear data — use
`LinearRegression`; scikit-learn's `Lasso` warns in the same situation.

## 7. Convergence

Cyclic sweeps over $j = 0, 1, \dots, d-1$. After each full sweep, stop when
the largest coefficient change during that sweep,
$\max_j \lvert w_j^{\text{new}} - w_j^{\text{old}}\rvert$, is below `tol`;
otherwise stop after `max_iter` sweeps, emit a `ConvergenceWarning`, and
record `n_iter_`. (scikit-learn uses a duality-gap criterion;
max-coefficient-change is the standard simple stopping rule and is
sufficient here.)

## 8. Algorithm (pseudocode)

```
Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4)

fit(X, y):
    if alpha < 0: raise ValueError
    if max_iter < 1: raise ValueError
    X, y = check_X_y(X, y)
    if fit_intercept:
        x_mean, y_mean = X.mean(0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
    else:
        Xc, yc = X, y
    n, d = Xc.shape
    w = zeros(d)
    z = (Xc ** 2).sum(0) / n              # z_j = ||x_j||^2 / n   (precomputed)
    r = yc - Xc @ w                        # residual, kept current below

    for sweep in 1 .. max_iter:
        max_change = 0
        for j in 0 .. d-1:
            if z[j] == 0: continue                         # constant feature
            rho_j   = Xc[:, j] @ r / n + w[j] * z[j]       # ρ_j
            w_j_new = soft_threshold(rho_j, alpha) / z[j]  # S(ρ_j, α) / z_j
            change  = w_j_new - w[j]
            if change != 0:
                r -= change * Xc[:, j]                     # restore r = yc - Xc w
                w[j] = w_j_new
                max_change = max(max_change, abs(change))
        n_iter_ = sweep
        if max_change < tol: break
    else:
        warn(ConvergenceWarning)

    coef_      = w
    intercept_ = y_mean - x_mean @ w   if fit_intercept else 0.0
    return self

soft_threshold(rho, alpha):
    if rho >  alpha: return rho - alpha
    if rho < -alpha: return rho + alpha
    return 0.0

predict(X):  check_is_fitted; validate n_features;  return X @ coef_ + intercept_
score(X, y): r2_score(y, predict(X))
```

Module-level `_soft_threshold(rho, alpha)` and `_lasso_objective(X, y, w, b,
alpha)` factor out the math for the tests, mirroring `_ridge_objective` in
`ridge.py`. There is no gradient-check tier — $J$ is not differentiable; the
**KKT check** in §9 is its correctness analogue.

There is no `solver` parameter: coordinate descent is the only method (unlike
`Ridge`/`LinearRegression`, which had a genuine normal-equation vs.
gradient-descent choice). scikit-learn's `Lasso` has no `solver` argument
either.

## 9. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | `_soft_threshold` matches its definition for values inside and outside $[-\alpha, \alpha]$, both signs, and at the boundary. |
| Analytic | A single standardised feature with a hand-computed $\rho$ recovers `coef_ ≈ S(ρ, α)`. |
| KKT / optimality | At the fitted $w$, with $c = \frac1n X_c^\top(y_c - X_c w)$: $c_j \approx \alpha\operatorname{sign}(w_j)$ for $w_j \ne 0$, and $\lvert c_j\rvert \le \alpha (1 + \varepsilon)$ for $w_j = 0$. (Replaces the gradient check.) |
| Reduction | `Lasso(alpha=0)` matches `LinearRegression` on well-conditioned seeded data. |
| Sparsity | A large enough $\alpha$ makes some `coef_` entries *exactly* `0.0`; past a further point all of them, with `intercept_ ≈ ȳ`. |
| Monotonicity | The number of non-zero coefficients is non-increasing as $\alpha$ grows; $\lVert w\rVert_1$ decreases. |
| Consistency | The objective $J$ is non-increasing across sweeps. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; `coef_` has shape `(d,)`; feature-count mismatch in `predict` raises; `alpha < 0` raises; `max_iter < 1` raises; repr round-trips. |
| Behavioral | On data whose true $w$ has many zeros, Lasso zeros most of the irrelevant features and beats OLS on held-out $R^2$. |
| Edge | `fit_intercept=False` on centered data; single feature; `max_iter=1` emits `ConvergenceWarning`; a constant feature gets `coef_` entry `0.0`. |
| Reference (`-m reference`) | `coef_` / `intercept_` / `predict` match `sklearn.linear_model.Lasso` (both run to a tight tolerance) within `atol=1e-4` on coefficients, `rtol=1e-4` on predictions. |

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  §3.4.2 (the lasso) and §3.8.6 (coordinate descent).
- Friedman, Hastie, Höfling, Tibshirani (2007), "Pathwise coordinate
  optimization", *Annals of Applied Statistics* 1(2).
- Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso",
  *JRSS B* 58(1).
