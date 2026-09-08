# Logistic regression (binary, MLE / cross-entropy)

The finalized derivation walkthrough for `scratchgrad.linear.LogisticRegression`,
written per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md) and mirrors
[`linear_regression.md`](linear_regression.md) and [`ridge.md`](ridge.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | design matrix — $n$ samples, $d$ features |
| $y$ | $(n,)$ | binary labels, encoded $\{0, 1\}$ internally |
| $w$ | $(d,)$ | weight vector (`coef_`) |
| $b$ | scalar | intercept (`intercept_`) |
| $\tilde{X} = [\mathbf{1} \;\; X]$ | $(n, d{+}1)$ | augmented design matrix |
| $\theta = [b, w]$ | $(d{+}1,)$ | folded parameter vector |
| $z = \tilde{X}\theta$ | $(n,)$ | logits (log-odds) |
| $p = \sigma(z)$ | $(n,)$ | predicted $P(y = 1 \mid x)$ |
| $S = \mathrm{diag}\big(p_i(1 - p_i)\big)$ | $(n, n)$ | Fisher / IRLS weights |
| $C$ | scalar $> 0$ | inverse L2 strength (`C`) |
| $D$ | $(d{+}1, d{+}1)$ | $\mathrm{diag}(0, 1, \dots, 1)$ — identity with a $0$ in the intercept slot |

**Model.** The probability of the positive class is a sigmoid of an affine
score:

$$P(y = 1 \mid x) = \sigma(w^\top x + b), \qquad
  \sigma(z) = \frac{1}{1 + e^{-z}}$$

The intercept is folded in by prepending a constant-1 column to $X$ exactly as
in [`ridge.md`](ridge.md), so $p = \sigma(\tilde{X}\theta)$. When
`fit_intercept=False` the augmentation is skipped, `intercept_` is fixed at
$0$, and $D = I_d$.

**Binary only.** This pass implements two-class logistic regression
(`plan.md` M1). `fit` requires `y` to hold exactly two distinct values,
stores them sorted as `classes_`, and maps the smaller to $0$ and the larger
to $1$. The multiclass softmax generalisation is deferred.

## 2. Objective — Bernoulli MLE → mean cross-entropy

Each label is a Bernoulli draw with parameter $p_i$:

$$P(y_i \mid x_i) = p_i^{\,y_i}\,(1 - p_i)^{\,1 - y_i}$$

Assuming samples are independent, the log-likelihood is

$$\ell(\theta) = \sum_{i=1}^{n}
  \big[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\big]$$

Maximising $\ell$ is minimising the **mean negative log-likelihood** (= the
cross-entropy loss). As in the other linear models we divide by $n$ so the
learning rate does not have to be re-tuned when $n$ changes, and add an
optional L2 penalty on the weights only:

$$\boxed{\;J(\theta) = \frac{1}{n}\left[\,
  \sum_{i=1}^{n}\Big(\mathrm{softplus}(z_i) - y_i z_i\Big)
  \;+\; \frac{1}{2C}\lVert w \rVert_2^2 \,\right]\;}$$

using the identity, for $p = \sigma(z)$,

$$-\big[y\log p + (1 - y)\log(1 - p)\big]
  = \log\!\big(1 + e^{z}\big) - y z
  = \mathrm{softplus}(z) - y z.$$

*Derivation of the identity.* $\log p = \log\sigma(z) = -\mathrm{softplus}(-z)$
and $\log(1 - p) = \log\sigma(-z) = -\mathrm{softplus}(z)$; substitute and
use $\mathrm{softplus}(z) - \mathrm{softplus}(-z) = z$.

**Numerical stability.** Writing the loss as $\mathrm{softplus}(z) - yz$
means the code never evaluates $\log$ of a probability that has rounded to
exactly $0$ or $1$. `softplus` itself is computed as
$\log(1 + e^{z}) = \mathrm{logaddexp}(0, z)$, whose standard implementation
subtracts the max before exponentiating — the same trick
`scratchgrad.utils.math.logsumexp` uses — so no `exp` overflows. The link
$p = \sigma(z)$ uses the branch-wise `scratchgrad.utils.math.sigmoid`.

**Convexity.** $\mathrm{softplus}$ is convex (its second derivative
$\sigma(z)(1 - \sigma(z)) > 0$), $-y_i z_i$ is linear, and
$\lVert w \rVert_2^2$ is convex — so $J$ is convex, and *strictly* convex
whenever the penalty is present ($C < \infty$) or $\tilde{X}$ has full column
rank. Its Hessian (§4) is positive semidefinite everywhere. So any stationary
point is the global minimum, and there is no dependence on the starting
iterate beyond the number of steps.

## 3. Gradient

Differentiate the data term through $z = \tilde{X}\theta$. Since
$\dfrac{\mathrm{d}}{\mathrm{d}z}\mathrm{softplus}(z) = \sigma(z)$,

$$\frac{\partial}{\partial z_i}\Big(\mathrm{softplus}(z_i) - y_i z_i\Big)
  = \sigma(z_i) - y_i = p_i - y_i,$$

and the chain rule with $\dfrac{\partial z}{\partial \theta} = \tilde{X}$ gives

$$\boxed{\;\nabla_\theta J = \frac{1}{n}\left[\,
  \tilde{X}^\top (p - y) \;+\; \frac{1}{C} D\theta \,\right]\;}$$

This is the same shape as the OLS gradient
$\frac{1}{n}\tilde{X}^\top(\hat{y} - y)$ — the *only* change is that the
prediction is $p = \sigma(\tilde{X}\theta)$ instead of $\hat{y} = \tilde{X}\theta$.
$D\theta$ is "$\theta$ with the intercept entry zeroed". For `penalty=None`
the $\frac{1}{C}D\theta$ term is dropped.

## 4. Hessian and the two solvers

Differentiating the gradient again, with
$\dfrac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$:

$$\nabla^2_\theta J = \frac{1}{n}\left[\,
  \tilde{X}^\top S\,\tilde{X} \;+\; \frac{1}{C} D \,\right],
  \qquad S = \mathrm{diag}\big(p_i(1 - p_i)\big) \succeq 0.$$

$\tilde{X}^\top S \tilde{X}$ is positive semidefinite (it is a Gram matrix in
the $\sqrt{S}$-weighted inner product), so $J$ is convex; adding
$\frac{1}{C}D$ makes it positive *definite* on the weight block whenever the
penalty is on. **There is no closed form** — $p$ depends on $\theta$
nonlinearly — so both solvers iterate.

### `solver="gd"` — batch gradient descent

$$\theta^{(t+1)} = \theta^{(t)} - \mathrm{lr}\cdot\nabla_\theta J(\theta^{(t)})$$

The direct continuation of the `LinearRegression` / `Ridge` gradient-descent
path. Initialise $\theta^{(0)} = \mathbf{0}$. Stop when
$\lVert \nabla_\theta J \rVert_\infty < \mathrm{tol}$, else after `max_iter`
steps (emitting `ConvergenceWarning`). Converges much faster on
comparably-scaled features; the estimator never rescales its inputs, so pass
data through `StandardScaler` first.

### `solver="newton"` — Newton–Raphson (= IRLS), the default

$$\theta^{(t+1)} = \theta^{(t)} - \big[\nabla^2_\theta J\big]^{-1}\nabla_\theta J$$

solved as `np.linalg.solve(H, g)` — no explicit inverse (`np.linalg` as a
linear-algebra primitive, `plan.md` §6). Each step is equivalently a
**weighted least-squares** fit with weights $S$, which is why this is called
*iteratively reweighted least squares*: expanding the update gives

$$\theta^{(t+1)} = \big(\tilde{X}^\top S \tilde{X} + \tfrac{1}{C}D\big)^{-1}
  \tilde{X}^\top S\, \big(\tilde{X}\theta^{(t)} + S^{-1}(y - p)\big),$$

a least-squares solve against the "working response" in the bracket. Newton
converges quadratically — typically 5–15 iterations — and has **no learning
rate to tune**, which is why it is the default (mirroring the "exact-ish
default + GD opt-in" split the other linear models already have). Stop when
the Newton decrement $\tfrac12\, g^\top H^{-1} g < \mathrm{tol}$, else after
`max_iter` steps.

### Separable data

If the two classes are linearly separable and there is **no penalty**
($C = \infty$), the MLE does not exist — $\lVert\theta\rVert \to \infty$ drives
the loss to $0$ but never attains it. The default `penalty="l2", C=1.0` keeps
$H$ positive definite and the solution finite and unique. With `penalty=None`
on separable data both solvers diverge (and Newton's solve may hit a singular
$H$) — a documented degenerate case, as in scikit-learn.

## 5. Predict and score

- `predict_proba(X)` → shape $(n, 2)$, columns $[\,1 - p,\; p\,]$
  (scikit-learn's column order: negative class first).
- `predict(X)` → `classes_[(p >= 0.5).astype(int)]`, i.e. threshold the
  positive-class probability at $0.5$ and map the $\{0, 1\}$ index back to the
  original label values.
- `score(X, y)` → `accuracy_score` (`scratchgrad.metrics`).

## 6. The `C` scaling and scikit-learn parity

scikit-learn's `LogisticRegression(penalty="l2", C=...)` minimises

$$\frac{1}{2}\lVert w \rVert_2^2 \;+\; C \sum_{i=1}^{n}
  \big(\mathrm{softplus}(z_i) - y_i z_i\big)$$

(the residual *sum*, no $\tfrac1n$). Our $J$ is that expression multiplied by
$\tfrac{1}{nC}$ — a positive constant, which does not move the minimiser — so
our **`C` equals scikit-learn's `C`** with no factor of $n$. This is the same
argument the `Ridge` doc makes for `alpha`, and as there, each estimator
deliberately matches *its own* scikit-learn counterpart's convention:
`Ridge` ↔ sklearn `Ridge`'s `alpha`, `Lasso` ↔ sklearn `Lasso`'s `alpha`,
`LogisticRegression` ↔ sklearn `LogisticRegression`'s `C`. The relationship is
$C = \tfrac{1}{2\alpha n}$ against a mean-form `alpha` convention, but `C` is
the genuine standard for this model, so that is what the API exposes.

`penalty=None` ↔ scikit-learn `penalty=None`: the $\tfrac{1}{C}$ term is
dropped from $J$, $g$, and $H$.

The parity test compares `coef_` / `intercept_` / `predict_proba` against
`sklearn.linear_model.LogisticRegression` (their default `lbfgs` vs. our
Newton, both run to a tight tolerance) at `rtol ≈ 1e-5`.

## 7. Algorithm (pseudocode)

```
LogisticRegression(C=1.0, penalty="l2", fit_intercept=True,
                   solver="newton", lr=0.1, max_iter=1000, tol=1e-6)

fit(X, y):
    if solver not in {"newton", "gd"}: raise ValueError
    if penalty not in {"l2", None}:    raise ValueError
    if C <= 0:                         raise ValueError
    if max_iter < 1:                   raise ValueError
    X, y = check_X_y(X, y)
    classes_ = unique(y);  if len(classes_) != 2: raise ValueError
    t = (y == classes_[1]).astype(float)          # labels -> {0, 1}

    Xt    = [1 | X] if fit_intercept else X
    mask  = [0, 1, ..., 1] if fit_intercept else [1, ..., 1]   # diag(D)
    invC  = 1 / C if penalty == "l2" else 0.0
    theta = zeros(Xt.shape[1])

    for it in 1 .. max_iter:
        p = sigmoid(Xt @ theta)
        g = (Xt.T @ (p - y) + invC * (mask * theta)) / n        # ∇J
        if solver == "gd":
            theta -= lr * g
            n_iter_ = it
            if max_abs(g) < tol: break
        else:  # newton
            S = p * (1 - p)
            H = (Xt.T @ (S[:, None] * Xt) + invC * diag(mask)) / n
            step = solve(H, g)
            theta -= step
            n_iter_ = it
            if 0.5 * (g @ step) < tol: break
    else:
        warn(ConvergenceWarning)

    if fit_intercept: intercept_, coef_ = theta[0], theta[1:]
    else:             intercept_, coef_ = 0.0, theta
    return self

predict_proba(X):  p = sigmoid(check_array(X) @ coef_ + intercept_); return [1-p, p]
predict(X):        return classes_[(predict_proba(X)[:, 1] >= 0.5).astype(int)]
score(X, y):       return accuracy_score(y, predict(X))
```

Module-level `_logistic_objective(X_aug, y, theta, inv_C, penalty_mask)`,
`_logistic_gradient(...)`, and `_logistic_hessian(X_aug, theta, inv_C,
penalty_mask)` factor out the math so the gradient- and Hessian-check tests
target them directly, mirroring `_ridge_objective` / `_ridge_gradient` in
`ridge.py`.

## 8. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | At $\theta = 0$, $p \equiv \tfrac12$, so $\nabla_\theta J = \frac1n \tilde{X}^\top(\tfrac12\mathbf{1} - y)$ — checked against a hand computation. |
| Gradient | $g = \frac1n(\tilde{X}^\top(p - y) + \frac1C D\theta)$ matches central finite differences of $J(\theta)$ (`tests/helpers/gradcheck.py`), with `penalty="l2"` and `penalty=None`. |
| Hessian | $H = \frac1n(\tilde{X}^\top S\tilde{X} + \frac1C D)$ matches finite differences of $g(\theta)$, column by column. |
| Convexity | $J$ is non-increasing across Newton iterations (refit with growing `max_iter`). |
| Solver consistency | `solver="gd"` (scaled features, many iterations) converges to `solver="newton"`. |
| Regularisation | `penalty=None` yields a larger $\lVert w \rVert_2$ than `penalty="l2", C=1.0` on the same data; both separate the training set. |
| Contract | `predict` before `fit` raises `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; `coef_` has shape `(d,)`; feature-count mismatch in `predict` raises; non-binary `y` raises; unknown `solver` / `penalty` raises; `C <= 0` raises; `max_iter < 1` raises; `predict_proba` rows sum to 1; `score` is accuracy; repr round-trips. |
| Behavioral | On `make_blobs(centers=2)` (near-separable) training accuracy $> 0.95$; on noisy `make_moons` accuracy beats the majority-class baseline by a clear margin. |
| Edge | `fit_intercept=False`; single feature; `max_iter=1` emits `ConvergenceWarning` and records `n_iter_`. |
| Reference (`-m reference`) | `coef_` / `intercept_` / `predict_proba` match `sklearn.linear_model.LogisticRegression` within `rtol ≈ 1e-5`. |

Plus `examples/logistic_regression.py` — seeded two-class blobs: fit, report
accuracy and the learned boundary, and sweep `C` to show coefficient norm and
accuracy trading off.

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd ed.,
  §4.4 (logistic regression) and §4.4.1 (fitting by IRLS).
- Bishop, *Pattern Recognition and Machine Learning*, §4.3.2–4.3.3 (logistic
  regression, iterative reweighted least squares).
- Murphy, *Machine Learning: A Probabilistic Perspective*, §8.3 (logistic
  regression, Newton's method).
