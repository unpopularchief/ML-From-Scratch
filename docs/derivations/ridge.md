# Ridge regression (L2-penalised least squares)

The finalized derivation walkthrough for `scratchgrad.linear.Ridge`, written
per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and mirrors
[`linear_regression.md`](linear_regression.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — $n$ samples, $d$ features |
| $y$ | $(n,)$ | regression targets |
| $w$ | $(d,)$ | weight vector (`coef_`) |
| $b$ | scalar | intercept (`intercept_`) |
| $\tilde{X} = [\mathbf{1} \;\; X]$ | $(n, d{+}1)$ | augmented design matrix |
| $\theta = [b, w]$ | $(d{+}1,)$ | folded parameter vector |
| $\alpha$ | scalar $\ge 0$ | regularisation strength (`alpha`) |
| $D$ | $(d{+}1, d{+}1)$ | $\operatorname{diag}(0, 1, \dots, 1)$ — identity with a $0$ in the intercept slot |

The model is unchanged from OLS: $\hat{y} = \tilde{X}\theta$, with the
intercept folded in by prepending a constant-1 column to $X$. When
`fit_intercept=False` the augmentation is skipped, `intercept_` is fixed at
$0$, and $D = I_d$.

Python reserves `lambda`, so the hyperparameter is named **`alpha`** — which
also matches `sklearn.linear_model.Ridge(alpha=...)` (see §6).

## 2. Objective — OLS plus an L2 penalty

$$J(\theta) = \frac{1}{n}\Big( \lVert \tilde{X}\theta - y \rVert_2^2
            \;+\; \alpha \lVert w \rVert_2^2 \Big)$$

Three things to note:

**The penalty is on $w$ only, never the intercept $b$.** Penalising $b$ would
make the fit depend on where the target's origin sits: add a constant to
every $y_i$ and the solution would shift in a way it should not. We write the
penalty as $\lVert w \rVert_2^2 = \theta^\top D\theta$ with
$D = \operatorname{diag}(0, 1, \dots, 1)$, so the intercept entry is excluded.

**The $\tfrac1n$ wraps both terms.** This keeps the data term equal to the
mean squared error (consistent with `LinearRegression`) *and* makes `alpha`
numerically identical to scikit-learn's — multiplying an objective by a
positive constant does not move its minimiser (§6).

**$J$ is convex, and strictly convex when $\alpha > 0$.** Its Hessian is

$$\nabla^2_\theta J = \frac{2}{n}\big( \tilde{X}^\top\tilde{X} + \alpha D \big),$$

which is positive semidefinite always, and positive *definite* whenever
$\alpha > 0$ — the $\alpha D$ term lifts every non-intercept eigenvalue by
$\alpha$. So for $\alpha > 0$ the minimiser is unique even when
$\tilde{X}$ is rank-deficient ($d + 1 > n$, or exactly collinear features).
That is the entire point of ridge.

## 3. Why ridge helps

- **Conditioning.** OLS needs $\tilde{X}^\top\tilde{X}$ invertible and
  well-conditioned. Near-collinear features make it nearly singular, so a
  tiny change in $y$ swings $w$ wildly. Adding $\alpha D$ lifts every
  non-intercept eigenvalue by $\alpha$, bounding the condition number.
- **Multicollinearity.** When two features are nearly redundant, OLS is free
  to place a huge $+M$ on one and $-M$ on the other. The $\lVert w \rVert_2^2$
  term makes that expensive, so ridge splits the weight between them instead.
- **Bias–variance.** Ridge deliberately trades a little bias (coefficients
  shrink toward $0$) for a large reduction in variance. For some $\alpha > 0$
  the variance reduction outweighs the added bias² and expected test error
  drops.
- **Shrinkage, not selection.** Coefficients move smoothly toward $0$ as
  $\alpha \to \infty$ but never become *exactly* $0$. Sparsity is Lasso's
  L1 penalty — a separate algorithm.

## 4. Closed form — the regularised normal equation

Expand the objective (dropping the constant $\tfrac1n y^\top y$ has no effect
on the argmin):

$$J(\theta) = \frac{1}{n}\Big( \theta^\top \tilde{X}^\top\tilde{X}\,\theta
  - 2\, y^\top \tilde{X}\theta + y^\top y
  + \alpha\, \theta^\top D\theta \Big)$$

Differentiate, using $\nabla_\theta(\theta^\top A\theta) = 2A\theta$ for
symmetric $A$ (both $\tilde{X}^\top\tilde{X}$ and $D$ are symmetric):

$$\nabla_\theta J = \frac{2}{n}\Big( \tilde{X}^\top(\tilde{X}\theta - y)
  + \alpha D\theta \Big)$$

Set the gradient to zero and multiply through by $\tfrac{n}{2}$:

$$\tilde{X}^\top\tilde{X}\,\theta + \alpha D\theta - \tilde{X}^\top y = 0$$

$$\boxed{\;(\tilde{X}^\top\tilde{X} + \alpha D)\,\theta = \tilde{X}^\top y\;}$$

This is the **regularised normal equation** — a ridge $\alpha$ added along the
diagonal (except the intercept entry), which is where the method's name comes
from. Textbook form is
$\theta = (\tilde{X}^\top\tilde{X} + \alpha D)^{-1}\tilde{X}^\top y$, but the
implementation uses `np.linalg.solve` on the system directly rather than
forming the inverse — same solution, better numerics. For $\alpha > 0$ the
left-hand matrix is symmetric positive definite, so the solve is always
well-posed. `np.linalg` here is a linear-algebra primitive, explicitly
allowed by `plan.md` §6.

**Ridge as OLS on augmented data.** An equivalent view:

$$\lVert \tilde{X}\theta - y \rVert_2^2 + \alpha \lVert w \rVert_2^2
  = \left\lVert
    \begin{bmatrix} \tilde{X} \\ \sqrt{\alpha}\,R \end{bmatrix}\theta
    - \begin{bmatrix} y \\ \mathbf{0}_d \end{bmatrix}
  \right\rVert_2^2,
  \qquad R = [\,\mathbf{0}\;\;I_d\,] \in \mathbb{R}^{d \times (d+1)}$$

i.e. ridge is plain least squares with $d$ extra pseudo-observations, each
pulling one weight toward $0$. The implementation solves the diagonal form
above (it states the "ridge on the diagonal" intuition most directly), but
the two are identical.

**Intercept handling matches scikit-learn exactly.** For any fixed $w$, the
$b$ minimising $J$ is $b^\star = \bar{y} - \bar{X}w$. Substituting it back
turns the objective into $\lVert y_c - X_c w \rVert_2^2 + \alpha\lVert w
\rVert_2^2$ on centered data — precisely what scikit-learn's `Ridge` solves
after centering $X$ and $y$. Augment-and-don't-penalise-$b$ and
center-then-solve give the same $\theta$.

## 5. Iterative — batch gradient descent

The `solver="gd"` path optimises the same objective with no linear solve.
From §4 the gradient at the current parameters is

$$g^{(t)} = \nabla_\theta J(\theta^{(t)})
  = \frac{2}{n}\Big( \tilde{X}^\top\big( \tilde{X}\theta^{(t)} - y \big)
  + \alpha D\theta^{(t)} \Big)$$

with the update $\theta^{(t+1)} = \theta^{(t)} - \mathrm{lr}\cdot g^{(t)}$.
$D\theta$ is just "$\theta$ with the intercept entry zeroed". Initialise
$\theta^{(0)} = \mathbf{0}$ (the problem is convex, so the starting point
affects only the iteration count). Stop when
$\lVert g^{(t)} \rVert_\infty < \mathrm{tol}$, or after `max_iter` steps —
in which case a `ConvergenceWarning` is emitted and `n_iter_` records how
many steps ran. As with OLS, gradient descent converges much faster on
comparably-scaled features and the estimator never rescales its inputs
(`plan.md` §1), so callers should pass data through `StandardScaler` first.

## 6. The `alpha` scaling and scikit-learn parity

scikit-learn's `Ridge(alpha)` minimises
$\lVert y - Xw \rVert_2^2 + \alpha \lVert w \rVert_2^2$ — the residual *sum*
of squares, not the mean. Our objective is that same expression times
$\tfrac1n$, so it has the **identical minimiser** and our `alpha` equals
scikit-learn's `alpha` with no factor of $n$. Both reduce to the normal
equation $(\tilde{X}^\top\tilde{X} + \alpha D)\theta = \tilde{X}^\top y$.
The parity test compares against
`sklearn.linear_model.Ridge(alpha=<same>, fit_intercept=<same>)` at
`rtol=1e-6` with the default solver on both sides.

## 7. `alpha = 0` reduces to OLS

At $\alpha = 0$ the penalty term vanishes and the regularised normal equation
becomes OLS's exactly, so `Ridge(alpha=0)` matches `LinearRegression`
coefficient-for-coefficient. One caveat: at $\alpha = 0$ with a
rank-deficient $\tilde{X}$ the `np.linalg.solve` system is singular, whereas
`LinearRegression`'s `np.linalg.lstsq` still returns the minimum-norm
solution. `Ridge(alpha=0)` on collinear data is a degenerate case — use
`LinearRegression` there. scikit-learn's `Ridge` warns in the same
situation.

## 8. Algorithm (pseudocode)

```
Ridge(alpha=1.0, fit_intercept=True, solver="normal",
      lr=0.01, max_iter=1000, tol=1e-6)

fit(X, y):
    if solver not in {"normal", "gd"}: raise ValueError
    if alpha < 0: raise ValueError
    X, y = check_X_y(X, y)
    Xt   = [1 | X] if fit_intercept else X
    mask = [0, 1, ..., 1] if fit_intercept else [1, ..., 1]   # diag(D)

    if solver == "normal":
        A     = Xt.T @ Xt + alpha * diag(mask)     # X̃ᵀX̃ + αD
        theta = solve(A, Xt.T @ y)                 # A θ = X̃ᵀy
    elif solver == "gd":
        if max_iter < 1: raise ValueError
        theta = zeros(Xt.shape[1])
        for t in 1 .. max_iter:
            resid = Xt @ theta - y                             # (n,)
            grad  = (2/n) * (Xt.T @ resid + alpha * (mask * theta))
            theta -= lr * grad
            n_iter_ = t
            if max_abs(grad) < tol: break
        else:
            warn(ConvergenceWarning)

    if fit_intercept:
        intercept_, coef_ = theta[0], theta[1:]
    else:
        intercept_, coef_ = 0.0, theta
    return self

predict(X):
    check_is_fitted(self, "coef_")
    validate X.shape[1] == coef_.shape[0]
    return X @ coef_ + intercept_

score(X, y):
    return r2_score(y, predict(X))
```

Module-level `_ridge_objective(X_aug, y, theta, alpha, penalty_mask)` and
`_ridge_gradient(...)` factor out the math so the gradient-check test targets
them directly, mirroring `_mse_objective` / `_mse_gradient` in
`linear_regression.py`.

## 9. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | A hand-solved small system with known $\alpha$ recovers `coef_` / `intercept_` to ~1e-10. |
| Analytic | `solver="normal"` matches a direct `np.linalg.solve` of $(\tilde{X}^\top\tilde{X} + \alpha D)\theta = \tilde{X}^\top y$ on a seeded problem. |
| Reduction | `Ridge(alpha=0)` equals `LinearRegression` (`coef_` / `intercept_`) on seeded data. |
| Shrinkage | $\lVert w \rVert_2$ decreases monotonically as $\alpha$ grows; $w \to 0$ and `intercept_` $\to \bar{y}$ as $\alpha \to \infty$. |
| Gradient | $g = \frac{2}{n}(\tilde{X}^\top(\tilde{X}\theta - y) + \alpha D\theta)$ matches central finite differences of $J(\theta)$ (`tests/helpers/gradcheck.py`), $\alpha > 0$. |
| Consistency | `solver="gd"` (scaled features, enough iterations) converges to `solver="normal"`. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; `coef_` has shape `(d,)`; feature-count mismatch in `predict` raises; unknown `solver` raises; `alpha < 0` raises; `max_iter < 1` raises (gd); repr round-trips. |
| Behavioral | On data with a near-duplicate feature column, held-out $R^2$ beats unregularised OLS. |
| Edge | `fit_intercept=False` on centered data; single feature; `max_iter=1` emits `ConvergenceWarning`. |
| Reference (`-m reference`) | `coef_` / `intercept_` / `predict` match `sklearn.linear_model.Ridge` within `rtol=1e-6`. |

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  §3.4.1 (ridge regression).
- Bishop, *Pattern Recognition and Machine Learning*, §3.1.4 (regularised
  least squares).
- Hoerl & Kennard (1970), "Ridge Regression: Biased Estimation for
  Nonorthogonal Problems", *Technometrics* 12(1).
