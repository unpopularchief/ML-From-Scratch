# Linear regression (ordinary least squares)

The finalized derivation walkthrough for `scratchgrad.linear.LinearRegression`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — $n$ samples, $d$ features |
| $y$ | $(n,)$ | regression targets |
| $w$ | $(d,)$ | weight vector (`coef_`) |
| $b$ | scalar | intercept (`intercept_`) |
| $\hat{y}$ | $(n,)$ | predictions |

**Model.** Each prediction is an affine function of the features:

$$\hat{y}_i = w^\top x_i + b = \sum_{j=1}^{d} w_j x_{ij} + b$$

**Intercept via augmentation.** Rather than carry $b$ as a separate term
through every equation, prepend a constant-1 column to $X$:

$$\tilde{X} = [\mathbf{1} \;\; X] \in \mathbb{R}^{n \times (d+1)}, \qquad
  \theta = \begin{bmatrix} b \\ w \end{bmatrix} \in \mathbb{R}^{d+1}$$

Then $\hat{y} = \tilde{X}\theta$. The first entry of the fitted $\theta$ becomes
`intercept_`, the rest become `coef_`. When `fit_intercept=False` the
augmentation is skipped and `intercept_` is fixed at $0$.

## 2. Objective

Ordinary least squares minimizes the residual sum of squares. We divide by
$n$ (mean squared error) so a gradient-descent learning rate does not have to
be re-tuned when the sample count changes:

$$J(\theta) = \frac{1}{n} \lVert \tilde{X}\theta - y \rVert_2^2
            = \frac{1}{n} \sum_{i=1}^{n} (\tilde{x}_i^\top \theta - y_i)^2$$

$J$ is convex in $\theta$ — it is quadratic with Hessian
$\frac{2}{n}\tilde{X}^\top\tilde{X} \succeq 0$ — so any stationary point is a
global minimum.

## 3a. Closed form — the normal equation

Expand the objective:

$$J(\theta) = \frac{1}{n}\left( \theta^\top \tilde{X}^\top \tilde{X} \theta
             - 2\, y^\top \tilde{X} \theta + y^\top y \right)$$

Differentiate, using $\nabla_\theta\,(\theta^\top A \theta) = 2A\theta$ for
symmetric $A$ and $\nabla_\theta\,(c^\top\theta) = c$:

$$\nabla_\theta J
  = \frac{2}{n}\left( \tilde{X}^\top \tilde{X}\, \theta - \tilde{X}^\top y \right)$$

Set the gradient to zero:

$$\tilde{X}^\top \tilde{X}\, \theta = \tilde{X}^\top y$$

This is the **normal equation**. Textbook form is
$\theta = (\tilde{X}^\top\tilde{X})^{-1}\tilde{X}^\top y$, but forming the
inverse explicitly is both slower and less numerically accurate than solving
the system directly. The implementation uses `np.linalg.lstsq`, which solves
the least-squares problem via SVD: it needs no invertibility assumption and
returns the minimum-norm solution when $\tilde{X}$ is rank-deficient (e.g.
$d + 1 > n$, or exactly collinear features). `np.linalg` here is a
linear-algebra primitive, explicitly allowed by `plan.md` §6 — it is not
implementing the ML algorithm, only the linear solve the derivation ends in.

## 3b. Iterative — batch gradient descent

The `solver="gd"` path optimizes the same objective with no linear solve.
From §3a the gradient at the current parameters is

$$g^{(t)} = \nabla_\theta J(\theta^{(t)})
         = \frac{2}{n}\, \tilde{X}^\top \big( \tilde{X}\theta^{(t)} - y \big)$$

with the update

$$\theta^{(t+1)} = \theta^{(t)} - \mathrm{lr}\cdot g^{(t)}$$

Initialize $\theta^{(0)} = \mathbf{0}$ (the problem is convex, so the
starting point affects only the iteration count, not the optimum). Stop when
$\lVert g^{(t)} \rVert_\infty < \mathrm{tol}$, or after `max_iter` steps —
in which case a `ConvergenceWarning` is emitted and `n_iter_` records how many
steps ran. Gradient descent converges much faster on roughly-equally-scaled
features; the estimator does **not** scale data internally (hyperparameters
and inputs are never mutated by `fit` — `plan.md` §1), so callers should pass
data through `StandardScaler` first.

## 4. Algorithm (pseudocode)

```
fit(X, y):
    X, y = check_X_y(X, y)
    Xt = [1 | X] if fit_intercept else X

    if solver == "normal":
        theta, *_ = lstsq(Xt, y)                  # minimizes ||Xt theta - y||^2
    elif solver == "gd":
        theta = zeros(Xt.shape[1])
        for t in 1 .. max_iter:
            resid = Xt @ theta - y                # (n,)
            grad  = (2/n) * (Xt.T @ resid)        # (d+1,)
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
    return check_array(X) @ coef_ + intercept_

score(X, y):
    return r2_score(y, predict(X))
```

Constructor: `LinearRegression(fit_intercept=True, solver="normal", lr=0.01,
max_iter=1000, tol=1e-6)`.

## 5. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-computed points on $y = 2x + 1$ recover `coef_ ≈ [2]`, `intercept_ ≈ 1` to ~1e-10. |
| Analytic | `solver="normal"` agrees with `np.linalg.lstsq` on the augmented matrix for a seeded random problem. |
| Analytic | Noise-free linear data → `r2_score == 1`, residuals ≈ 0. |
| Gradient | $g = \frac{2}{n}\tilde{X}^\top(\tilde{X}\theta - y)$ matches central finite differences of $J(\theta)$ (`tests/helpers/gradcheck.py`). |
| Consistency | `solver="gd"` (enough iterations, scaled features) converges to within tolerance of `solver="normal"`. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; `coef_` has shape `(d,)`; feature-count mismatch in `predict` raises. |
| Behavioral | On `make_regression` with modest noise, held-out `score` exceeds a threshold. |
| Edge | `fit_intercept=False` on centered data; single feature; `max_iter=1` emits `ConvergenceWarning`; unknown `solver` raises. |
| Reference (`-m reference`) | `coef_` / `intercept_` / `predict` match `sklearn.linear_model.LinearRegression` within `rtol=1e-6`. |

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  §3.2 (linear regression models and least squares).
- Bishop, *Pattern Recognition and Machine Learning*, §3.1.
