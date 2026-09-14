# Principal Component Analysis

The finalized walkthrough for `scratchgrad.decomposition.PCA`, written per
`plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | data — unsupervised, no $y$ |
| $\bar x$ | $(d,)$ | feature-wise mean (`mean_`) |
| $X_c$ | $(n, d)$ | centered data, $X_c = X - \bar x$ |
| $C$ | $(d, d)$ | covariance matrix, $C = \frac{1}{n-1} X_c^\top X_c$ |
| $k$ | scalar $\le d$ | number of components to keep (`n_components`) |
| $w_j$ | $(d,)$ | $j$-th principal direction, $\lVert w_j \rVert = 1$ |
| $W$ | $(d, k)$ | $[w_1, \dots, w_k]$ — `components_.T` |
| $\lambda_j$ | scalar $\ge 0$ | variance captured by $w_j$, i.e. eigenvalue of $C$ |

## 2. The objective — two equivalent forms

**Variance maximization.** Find an orthonormal direction $w$ that
maximises the variance of the data projected onto it:

$$\max_{w:\, \lVert w \rVert = 1} \mathrm{Var}(X_c w) = \max_{w} w^\top C w$$

**Reconstruction-error minimization.** Find an orthonormal $W \in
\mathbb{R}^{d\times k}$ minimising the squared error of reconstructing
each point from its $k$-dimensional projection:

$$\min_{W:\, W^\top W = I} \sum_{i=1}^n \lVert x_i - \bar x - WW^\top(x_i - \bar x) \rVert^2$$

**These are the same problem.** For any orthonormal $w$, Pythagoras
decomposes each centered point into the part captured by $w$ and the
orthogonal residual:

$$\lVert x_i - \bar x \rVert^2 = \underbrace{(w^\top(x_i - \bar x))^2}_{\text{captured}} + \underbrace{\lVert \text{residual}_i \rVert^2}_{\text{reconstruction error}}$$

The left side is fixed (it doesn't depend on $w$), so maximising the
captured term summed over $i$ (= the projected variance, up to the
constant $n-1$) is *exactly* minimising the residual term summed over
$i$ (= the reconstruction error). The derivation below works from the
variance form — it reaches an eigenproblem in one Lagrangian step.

## 3. Derivation — Lagrangian to eigenproblem

**First direction.** Maximise $w^\top C w$ subject to $w^\top w = 1$:

$$\mathcal{L}(w, \lambda) = w^\top C w - \lambda(w^\top w - 1)$$

$$\frac{\partial \mathcal{L}}{\partial w} = 2Cw - 2\lambda w = 0 \quad\Longrightarrow\quad Cw = \lambda w$$

So the maximiser must be an eigenvector of $C$. The objective's value
*at* an eigenvector is $w^\top C w = w^\top(\lambda w) = \lambda$ — so
among all eigenvectors, the maximising choice is the one with the
**largest eigenvalue** $\lambda_1$.

**Subsequent directions.** $C$ is symmetric and positive
semi-definite ($v^\top C v = \mathrm{Var}(X_c v) \ge 0$ for any $v$), so
by the spectral theorem it has a complete set of $d$ orthonormal
eigenvectors. Adding the constraint $w_2 \perp w_1$ to the same
maximisation and repeating the Lagrangian argument (now with two
multipliers, one per constraint) shows $w_2$ must also be an eigenvector
of $C$, and among those orthogonal to $w_1$, the maximiser is the one
with the next-largest eigenvalue $\lambda_2$. Induction gives the full
ordering: $w_1, \dots, w_d$ are the eigenvectors of $C$ sorted by
eigenvalue $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$, and
keeping the top $k$ is the best possible rank-$k$ answer to *both* forms
of §2 simultaneously (Eckart–Young, specialised to a symmetric matrix).

**Conclusion:** `components_` are the top-$k$ eigenvectors of $C$;
`explained_variance_` is the corresponding $\lambda_j$ (the actual
variance of the data along that direction); `explained_variance_ratio_`
is $\lambda_j / \sum_{i=1}^d \lambda_i$.

## 4. Computing it — SVD of $X_c$, not `eigh(C)`

Forming $C = X_c^\top X_c / (n-1)$ explicitly squares the condition
number of the data (a direction with singular value $\sigma$ in $X_c$
becomes $\sigma^2$ in $C$), which can lose precision for
poorly-conditioned or high-dimensional data. Instead, take the SVD of
the centered data directly:

$$X_c = U \Sigma V^\top, \qquad U \in \mathbb{R}^{n\times r},\; \Sigma \in \mathbb{R}^{r\times r},\; V \in \mathbb{R}^{d\times r}$$

($r = \min(n, d)$, `full_matrices=False`). Substituting:

$$C = \frac{X_c^\top X_c}{n - 1} = \frac{(U\Sigma V^\top)^\top(U\Sigma V^\top)}{n-1} = \frac{V\Sigma U^\top U \Sigma V^\top}{n-1} = V \frac{\Sigma^2}{n-1} V^\top$$

using $U^\top U = I$. This is an eigendecomposition of $C$ ($C v_j =
\frac{\sigma_j^2}{n-1} v_j$) read directly off the SVD, with **no need
to ever form $C$**. So:

- `components_` = rows of $V^\top$ (i.e. columns of $V$), top $k$ by
  singular value.
- `singular_values_` = $\Sigma$'s top $k$ diagonal entries.
- `explained_variance_` = $\Sigma^2 / (n-1)$, top $k$.

This is what scikit-learn's default `svd_solver="full"` does too, and
for the identical numerical reason. It is allowed under `plan.md` §6:
`np.linalg.svd` is a permitted array/linear-algebra primitive — the
*algorithm* (§3's argument that variance-maximising directions are
covariance eigenvectors) is still derived by hand above, not outsourced
to the library call.

## 5. Sign ambiguity

If $w$ is a unit eigenvector of $C$ with eigenvalue $\lambda$, so is
$-w$ — the SVD gives no guarantee about which sign it returns, and two
mathematically-identical fits (e.g. re-running with a duplicated row
order) can silently flip signs. Fixed deterministically, matching
scikit-learn: for each row of `components_`, if its largest-magnitude
entry is negative, negate the whole row (and the corresponding column of
`U`, to keep `transform` consistent). This makes `components_`
reproducible and lets the equivalence test in §6 compare directions
without a spurious sign mismatch.

## 6. The "ship two implementations" requirement

`plan.md` §6 calls PCA out by name: *"Where a decomposition is the
lesson... ship a from-scratch version (power iteration, QR) alongside
the `np.linalg` one and add a test asserting they agree."* Alongside the
SVD-based `fit` above, `_power_iteration_pca` computes the same
`components_`/`explained_variance_` via classic **power iteration with
deflation** — an independent algorithm on the *same* $C$, not another
call to a library decomposition:

1. Form $C$ explicitly (fine here — this helper exists to be readable
   and cross-checked, not to be the fast/stable production path).
2. To find the top eigenvector: start from a random unit vector $v$,
   repeatedly apply $v \leftarrow Cv / \lVert Cv \rVert$. Since $C$ is
   symmetric, expanding $v_0$ in $C$'s eigenbasis shows each application
   scales the $\lambda_1$ component by $\lambda_1$ and every other
   component by a strictly smaller $\lambda_j$, so the $\lambda_1$
   direction dominates geometrically and $v$ converges to $w_1$ (up to
   sign) as long as $v_0$ isn't exactly orthogonal to it and
   $\lambda_1 > \lambda_2$.
3. **Deflation:** having found $(\lambda_1, w_1)$, subtract its
   contribution — $C \leftarrow C - \lambda_1 w_1 w_1^\top$ — which zeros
   out the $\lambda_1$ eigenvalue while leaving every other eigenvector
   and eigenvalue of $C$ unchanged (they're orthogonal to $w_1$, so the
   subtracted rank-1 term acts as zero on them). Repeat power iteration
   on the deflated matrix to get $(\lambda_2, w_2)$, and so on for $k$
   components.

This is tested (not shipped as a `fit` option) to produce the same
`components_` (up to sign) and `explained_variance_` as the SVD path —
that agreement *is* the lesson, mirroring the `nn`/`autograd` duplication
described in `plan.md` §1.

## 7. Scope

Keeping to the project's stated minimalism (`plan.md` §8 mistake 2):

- **`n_components: int | None` only.** No float variance-ratio
  threshold (e.g. `n_components=0.95`) and no `"mle"` — both are
  convenience features layered on top of the same eigenvalues, not new
  math; add them later only if a concrete need shows up.
- **No `whiten`.** Rescaling each component to unit variance is a
  solver-level convenience for downstream estimators, not part of the
  PCA derivation itself.
- **No `svd_solver` parameter.** Always the full `np.linalg.svd` — no
  randomized/truncated solver. Mirrors `KMeans` having no `algorithm`
  parameter: an acceleration structure is out of scope, not a
  statistical choice.
- **`transform`/`inverse_transform` only** — no `fit_transform`
  shortcut beyond calling both (matches the project's other
  transformers).

## 8. Algorithm (pseudocode)

```
PCA(n_components=None)

fit(X):
    X = check_array(X)
    n, d = X.shape
    k = n_components if n_components is not None else min(n, d)
    require 1 <= k <= min(n, d)

    mean_ = X.mean(axis=0)
    Xc = X - mean_
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # S has min(n,d) entries, descending
    _fix_signs(U, Vt)                                    # largest-|entry| of each row of Vt -> positive

    components_ = Vt[:k]
    singular_values_ = S[:k]
    explained_variance_ = S**2 / (n - 1)                 # ALL min(n,d) of them, for the ratio's denominator
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_[:k] / total_var
    explained_variance_ = explained_variance_[:k]
    mean_, components_, ... stored
    return self

transform(X):          (X - mean_) @ components_.T
inverse_transform(Xt):  Xt @ components_ + mean_
fit_transform(X):       fit(X).transform(X)

_fix_signs(U, Vt):
    for each row i of Vt:
        if Vt[i, argmax(abs(Vt[i]))] < 0:
            Vt[i] *= -1
            U[:, i] *= -1

_power_iteration_pca(X, k, n_iter, rng):    # test-only, cross-checked against fit()
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (n - 1)
    components, eigenvalues = [], []
    for _ in range(k):
        v = rng.normal(size=d); v /= ||v||
        for _ in range(n_iter):
            v = C @ v; v /= ||v||
        lam = v.T @ C @ v
        components.append(v); eigenvalues.append(lam)
        C -= lam * outer(v, v)                            # deflation
    return stack(components), array(eigenvalues)
```

## 9. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | A hand-built 2D dataset with an obvious principal axis (points scattered tightly along a 45° line, wider spread orthogonal to it) — `components_`/`explained_variance_` match a hand computation of $C$'s eigendecomposition. |
| Analytic | Diagonal covariance by construction (independent-per-axis data): `components_` recover the coordinate axes themselves (up to sign/order), `explained_variance_` recovers the per-axis variances directly. |
| Orthonormality | `components_ @ components_.T` is the identity matrix, for every `n_components`. |
| Objective equivalence | For a fitted model, the projected variance ($\sum_j$ `explained_variance_`) plus the mean squared reconstruction error (from `inverse_transform`) equals the total variance of $X$ — the §2 Pythagorean identity, checked numerically. |
| Independent-algorithm equivalence | `_power_iteration_pca` and `fit`'s SVD path produce the same `components_` (up to sign, `np.testing.assert_allclose` after a sign-alignment step) and the same eigenvalues, on several random datasets — the §6 requirement. |
| Explained variance ratio | Sums to 1 when `n_components == min(n_samples, n_features)`; is non-increasing in $j$; each ratio is in $[0, 1]$. |
| Determinism | Same data, same `n_components` → identical `components_`/`explained_variance_` across repeated fits (SVD has no RNG; sign-fixing removes the one source of nondeterminism `np.linalg.svd` could otherwise introduce). |
| Contract | `transform`/`inverse_transform` before `fit` raise `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; feature-count mismatch in `transform` raises; `n_components < 1` or `> min(n_samples, n_features)` raises; `repr` round-trips. |
| Reference (`-m reference`) | Exact parity with `sklearn.decomposition.PCA` (deterministic on both sides, no RNG involved anywhere in `svd_solver="full"`): same `components_` after sign-alignment, same `explained_variance_`/`explained_variance_ratio_`/`singular_values_`, same `transform` output. |
| Edge | `n_components=1`; already mean-zero input (`mean_` still computed and subtracted, near-zero numerically); a constant (zero-variance) feature (its eigenvalue is ~0, no crash, no NaN); `n_features=1`. |

Plus `examples/pca.py` — seeded, runnable: fit PCA on a correlated 2D
Gaussian blob and report how much variance 1 component captures;
project a higher-dimensional dataset (e.g. `make_blobs` with several
features) down to 2 components and report the reconstruction error
against keeping all components.

## References

- Pearson, K. (1901), "On Lines and Planes of Closest Fit to Systems of
  Points in Space", *Philosophical Magazine* 2(11).
- Hotelling, H. (1933), "Analysis of a Complex of Statistical Variables
  into Principal Components", *Journal of Educational Psychology* 24.
- Eckart, C. & Young, G. (1936), "The Approximation of One Matrix by
  Another of Lower Rank", *Psychometrika* 1(3).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §14.5.
- Golub, G. & Van Loan, C., *Matrix Computations*, 4th ed., §8.2 (power
  iteration and deflation), §8.6 (SVD).
