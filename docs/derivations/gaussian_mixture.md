# Gaussian Mixture Model (EM)

The finalized walkthrough for `scratchgrad.cluster.GaussianMixture`, written
per `plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | data to cluster — unsupervised, no $y$ |
| $K$ | scalar $\ge 1$ | number of mixture components (`n_components`) |
| $z_i$ | scalar $\in \{1,\dots,K\}$ | **latent, unobserved** component label of point $i$ |
| $\pi_k$ | scalar, $\sum_k \pi_k = 1$ | mixing weight $P(z_i = k)$ (`weights_`) |
| $\mu_k$ | $(d,)$ | mean of component $k$ (`means_`) |
| $\Sigma_k$ | shape depends on `covariance_type` | covariance of component $k$ (`covariances_`) |
| $\gamma_{ik}$ | scalar $\in [0,1]$, $\sum_k \gamma_{ik}=1$ | responsibility, $P(z_i=k \mid x_i)$ — the soft assignment |
| $N_k$ | scalar | $\sum_i \gamma_{ik}$, effective number of points "belonging" to component $k$ |

This is the direct latent-variable generalization of `GaussianNB`'s
per-class Gaussian model: there, $z_i$ (the class) is **observed** in
training; here it is not, which is the entire source of the difficulty
below. It is also the soft, probabilistic generalization of `KMeans` — see
§5.

## 2. The objective: why this isn't a closed-form MLE

Each point's marginal density, after summing out the latent $z_i$:

$$p(x_i) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x_i; \mu_k, \Sigma_k)$$

and the log-likelihood of the whole dataset:

$$\ell(\pi,\mu,\Sigma) = \sum_{i=1}^n \log \left[\sum_{k=1}^K \pi_k\, \mathcal{N}(x_i; \mu_k, \Sigma_k)\right]$$

Contrast with GaussianNB, where the class is observed and the sum is gone
before the log is taken, so $\partial \ell/\partial \mu_{ck} = 0$ has a
closed form. Here the **log of a sum** has no such closed form: setting
$\partial \ell / \partial \mu_k = 0$ leaves $\mu_k$ entangled with every
other component's parameters through the normalizing sum inside the log.
This is the same "no closed form" situation `LogisticRegression` hits with
its NLL, but the fix here is different — not gradient descent, but
Expectation-Maximization (Dempster, Laird, Rubin, 1977).

## 3. EM as coordinate ascent on a lower bound

**The trick.** For *any* distribution $q_i(\cdot)$ over $z_i$, Jensen's
inequality ($\log$ is concave) gives, for every point $i$:

$$\log \sum_k \pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)
  = \log \sum_k q_i(k) \frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{q_i(k)}
  \ge \sum_k q_i(k) \log \frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{q_i(k)}$$

Summing over $i$ defines the **evidence lower bound** (ELBO),
$\mathrm{ELBO}(q,\theta) \le \ell(\theta)$ for every choice of
$q = \{q_i\}$, with $\theta = (\pi,\mu,\Sigma)$. Equality holds iff
$q_i(k) = P(z_i=k\mid x_i;\theta)$ exactly (a standard Jensen-equality
argument: the ratio inside the log must be constant in $k$, which is
precisely the posterior).

EM alternates two exact maximizations of the ELBO — the same **block
coordinate ascent** structure as `KMeans`'s block coordinate *descent* on
$J$ (`docs/derivations/kmeans.md` §3), with the discrete assignment block
made *soft* (a distribution over $k$, not a single argmin).

### E-step — maximize over $q$, holding $\theta$ fixed

The bound is tight exactly at the posterior, so the maximizing $q$ is the
**responsibility**:

$$\gamma_{ik} \;=\; P(z_i = k \mid x_i; \theta)
  \;=\; \frac{\pi_k\, \mathcal{N}(x_i;\mu_k,\Sigma_k)}
             {\sum_{j=1}^K \pi_j\, \mathcal{N}(x_i;\mu_j,\Sigma_j)}$$

— Bayes' rule, exactly as in GaussianNB's posterior, except $\pi_k$ here
is *learned* rather than a fixed prior, and there is no observed label to
check the answer against.

### M-step — maximize over $\theta$, holding $q = \gamma$ fixed

With $\gamma$ fixed, the ELBO's $\theta$-dependent part is the **expected
complete-data log-likelihood**:

$$Q(\theta) = \sum_i \sum_k \gamma_{ik} \left[\log \pi_k + \log \mathcal{N}(x_i;\mu_k,\Sigma_k)\right]$$

This *does* separate cleanly (no log-of-sum — the sum over $k$ is now
*outside* every log), so each parameter has a closed-form maximizer:

**Weights** (Lagrange multiplier $\lambda$ for the constraint $\sum_k\pi_k=1$):
$$\frac{\partial}{\partial \pi_k}\left[\sum_i \gamma_{ik}\log\pi_k + \lambda\Big(\sum_k \pi_k - 1\Big)\right] = \frac{N_k}{\pi_k} + \lambda = 0
\;\Longrightarrow\; \pi_k = -\frac{N_k}{\lambda}$$
Summing both sides over $k$ ($\sum_k \pi_k = 1$, $\sum_k N_k = n$) gives
$\lambda = -n$, hence

$$\pi_k \leftarrow \frac{N_k}{n}, \qquad N_k := \sum_{i=1}^n \gamma_{ik}$$

**Means** (differentiate the Gaussian log-density term, $\Sigma_k^{-1}$ cancels):
$$\frac{\partial Q}{\partial \mu_k} = \sum_i \gamma_{ik}\, \Sigma_k^{-1}(x_i - \mu_k) = 0
\;\Longrightarrow\;
\mu_k \leftarrow \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\, x_i$$

the responsibility-weighted mean — the direct soft-assignment generalization of
KMeans's per-cluster mean (§5).

**Covariances** — the standard weighted-Gaussian-MLE result (differentiating
$Q$ with respect to the precision $\Sigma_k^{-1}$ using
$\partial \log\det A/\partial A = A^{-1}$ and $\partial\, x^\top A x/\partial A = xx^\top$,
then inverting back): the responsibility-weighted, mean-centered second moment,

$$\Sigma_k \leftarrow \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^\top$$

for `covariance_type="full"`. §9 works out the analogous closed form for
each of the other three structures — each is this same formula restricted
to a smaller parameter family, and each maximizer is that restriction's
weighted MLE.

## 4. Monotone increase and termination

$$\ell(\theta^{(t)}) = \mathrm{ELBO}(\gamma^{(t)}, \theta^{(t)})
  \le \mathrm{ELBO}(\gamma^{(t)}, \theta^{(t+1)})
  \le \mathrm{ELBO}(\gamma^{(t+1)}, \theta^{(t+1)})
  = \ell(\theta^{(t+1)})$$

(first equality: the E-step made the bound tight at $\theta^{(t)}$; first
inequality: the M-step re-maximizes over $\theta$ with $\gamma^{(t)}$
fixed; second inequality: ELBO $\le \ell$ always, by Jensen; final
equality: the *next* E-step re-tightens the bound.) So the observed-data
log-likelihood $\ell$ is **non-decreasing every iteration** — the EM
analogue of KMeans §4's inertia argument, with "decrease $J$" replaced by
"increase $\ell$." Like Lloyd's algorithm, this guarantees convergence to
a **local** maximum only, never a global one — hence `n_init` (§7) matters
here for exactly the reason it matters for KMeans.

## 5. The KMeans limit (a concrete, not just narrative, connection)

Fix $\pi_k = 1/K$ and let every $\Sigma_k = \sigma^2 I$ with the *same*
$\sigma^2 \to 0^+$. The responsibility becomes

$$\gamma_{ik} = \frac{\exp(-\lVert x_i-\mu_k\rVert^2/2\sigma^2)}{\sum_j \exp(-\lVert x_i-\mu_j\rVert^2/2\sigma^2)}$$

a softmax over negative squared distances with an inverse-temperature
$1/\sigma^2 \to \infty$: it sharpens to a **one-hot indicator of the
nearest centroid** — exactly KMeans's hard assignment step. The M-step's
$\mu_k \leftarrow \sum_i\gamma_{ik}x_i / N_k$ then becomes exactly KMeans's
cluster mean. `GaussianMixture` is what you get by *not* taking that
limit: it keeps the soft assignment and additionally learns each
cluster's own shape (`covariance_type="full"`/`"diag"`) and size
(`weights_`, not assumed uniform) — a genuine generative model of the
data, capable of `sample()`-ing new points, unlike KMeans's purely
partition-defining objective.

## 6. Initialization

EM needs a starting point before the first E-step — in practice, an
initial *responsibility* matrix $\gamma^{(0)}$, from which an M-step
produces the first $(\pi,\mu,\Sigma)$.

**`init_params="random"`.** Draw $\gamma^{(0)}_{i\cdot} \sim \mathrm{Uniform}(0,1)^K$
independently per row, then normalize each row to sum to 1. No structure
at all — simplest, but like KMeans's `init="random"`, reliably slower to
converge and more sensitive to a bad draw.

**`init_params="kmeans"` (the default, matching scikit-learn).** Run this
project's own `KMeans(n_clusters=n_components)` once (§ `cluster/kmeans.py`)
to get hard labels, then set $\gamma^{(0)}$ to the one-hot encoding of
those labels. The first M-step then produces exactly the per-cluster mean
and (co)variance of the KMeans partition as the starting Gaussians — a far
better starting point than random responsibilities, since it is already a
local optimum of the *hard*-assignment relaxation of this same problem
(§5). This is a genuine cross-module reuse: `KMeans` is a real dependency
of `GaussianMixture`'s default path, not just a narrative connection.

## 7. `n_init` restarts

Identical in spirit to KMeans §5's `n_init`: EM converges to a local
optimum that depends on where $\gamma^{(0)}$ started. `n_init` independent
restarts are each run to convergence (or `max_iter`); the run with the
**highest** final log-likelihood is kept (KMeans keeps the *lowest*
inertia — the sign flips because log-likelihood is "higher is better,"
inertia is "lower is better," but it is the same "best of several local
optima" defense).

## 8. Convergence, `reg_covar`, and numerical stability

**Stopping rule.** Unlike KMeans (which tracks centroid shift), EM has a
natural monotone scalar to track directly: the **average per-sample
log-likelihood**, $\bar\ell^{(t)} = \frac1n \sum_i \log p(x_i)$, computed
for free as a byproduct of the E-step's normalizer. Stop when
$\bar\ell^{(t)} - \bar\ell^{(t-1)} < \texttt{tol}$ or `max_iter` is
reached — matching scikit-learn's own criterion exactly (averaging makes
`tol` roughly independent of `n`, the same motivation as KMeans's
variance-scaled `tol`, applied to a quantity that's already an average).

**`reg_covar`.** A component that captures very few points, or points
that are nearly collinear, can produce a covariance matrix that is
singular or near-singular — inverting it (needed for the E-step's density)
blows up or raises a linear-algebra error. `reg_covar` (default `1e-6`,
matching scikit-learn) is added to the diagonal of every $\Sigma_k$ after
each M-step, exactly analogous to `GaussianNB`'s `var_smoothing`/`epsilon_`
variance floor — cheap insurance that costs nothing when the covariance is
already well-conditioned.

**Log-space computation.** As in `GaussianNB`, the E-step is computed in
log-space (`_estimate_log_gaussian_prob` + `scratchgrad.utils.math.logsumexp`)
rather than forming raw densities and normalizing — a raw
$\mathcal N(x;\mu,\Sigma)$ density underflows to exactly `0.0` in high
dimensions or far tails long before the responsibilities actually vanish,
which is the multivariate-Gaussian version of the same
`logaddexp`/`logsumexp` stability argument used throughout this project.

## 9. The four `covariance_type` structures

All four share the E-step formula of §3 and the same $\mu_k$ update; they
differ only in *how much freedom* $\Sigma_k$ is given, i.e. what family
the M-step's weighted-Gaussian-MLE is restricted to. Each is a genuine,
separately-motivated modeling assumption, not an implementation detail —
this is the concrete generalization of "GaussianNB assumes each feature is
independent within a class" (that assumption *is* `covariance_type="diag"`
below, applied per-class instead of per-mixture-component).

- **`"full"`** — no restriction, $\Sigma_k \in \mathbb{R}^{d\times d}$ SPD,
  one independent matrix per component:
  $$\Sigma_k = \frac{1}{N_k}\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$
  Most expressive ($K\cdot d(d+1)/2$ free covariance parameters); can
  model correlated features and elliptical clusters at any orientation.

- **`"diag"`** — force $\Sigma_k$ diagonal, i.e. assume features are
  conditionally independent *within* a component (the GaussianNB
  assumption, applied per mixture component instead of per observed
  class):
  $$\sigma^2_{kj} = \frac{1}{N_k}\sum_i \gamma_{ik}(x_{ij}-\mu_{kj})^2, \quad j=1,\dots,d$$
  Axis-aligned elliptical clusters only. $K\cdot d$ parameters.

- **`"spherical"`** — further restrict `"diag"` to a single shared scalar
  variance across all $d$ features of one component (isotropic — the same
  spread in every direction, but the component's own scale is still
  learned, unlike KMeans which has no scale at all):
  $$\sigma^2_k = \frac{1}{N_k\, d}\sum_i \gamma_{ik}\lVert x_i-\mu_k\rVert^2$$
  This is exactly the §5 KMeans limit *without* taking $\sigma\to 0$ — a
  genuinely useful middle ground: circular/spherical clusters of
  *different, learned* sizes. $K$ parameters.

- **`"tied"`** — back to a full (correlated) covariance, but a **single**
  $\Sigma$ shared by every component (only the means and weights
  distinguish clusters):
  $$\Sigma = \frac{1}{n}\sum_k \sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$
  Note the normalizer is $n$ (total points), not $N_k$ — this is the
  pooled within-component scatter, the mixture-model analogue of Linear
  Discriminant Analysis's shared covariance assumption. $d(d+1)/2$
  parameters total, independent of $K$.

Each is its own weighted-MLE derivation (a constrained special case of the
`"full"` result — restricting the maximization to matrices of the stated
form), not an approximation of `"full"`.

**Numerical implementation, all four:** every log-density is computed via
a **Cholesky factor** of $\Sigma_k$ (or the shared $\Sigma$ for `"tied"`),
$\Sigma_k = L_kL_k^\top$, giving $\log\det\Sigma_k = 2\sum_j \log (L_k)_{jj}$
and the Mahalanobis term $(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k) = \lVert
L_k^{-1}(x-\mu_k)\rVert^2$ via a triangular solve — the same
numerically-preferred approach `LogisticRegression`'s Newton step and
`Ridge`'s normal equation use (`np.linalg.solve` instead of forming an
explicit inverse), generalized from a linear system to a covariance
factorization. A `np.linalg.LinAlgError` from a non-positive-definite
$\Sigma_k$ (can still happen if `reg_covar` is set to `0` and a component
collapses onto too few points) is caught and re-raised as a `ValueError`
naming `reg_covar` as the fix — mirroring scikit-learn's own message
here, since "silently produce `nan`" would be far worse.

## 10. Beyond fit/predict: the generative extras

Because $p(x)$ and $P(z\mid x)$ are both explicit, fitted densities (not
just a partition), several methods fall out for free that KMeans has no
analogue for:

- **`predict_proba(X)`** — the responsibilities $\gamma_{ik}$ themselves
  (a real posterior, not a heuristic soft-max-of-distance the way one
  might retrofit onto KMeans).
- **`predict(X)`** — $\arg\max_k \gamma_{ik}$, ties to the lowest index.
- **`score_samples(X)`** — $\log p(x_i)$ per row (the E-step's normalizer,
  before any responsibility is even formed).
- **`score(X)`** — $\frac1n\sum_i \log p(x_i)$, the mean of the above —
  scikit-learn's convention and this project's own convergence criterion
  (§8), so `score` is literally the same quantity `fit` was maximizing.
- **`sample(n_samples)`** — draw component counts from
  $\mathrm{Multinomial}(n\_samples, \pi)$, then draw that many points per
  component from $\mathcal N(\mu_k,\Sigma_k)$ via
  $x = \mu_k + L_k z,\; z\sim\mathcal N(0,I)$ (the same Cholesky factor as
  §9's log-density, reused for generation instead of evaluation). Returns
  `(X, y)` with `y` the generating component index — the concrete payoff
  of GMM being a real generative model, which KMeans's pure partitioning
  objective never was.
- **`bic(X)` / `aic(X)`** — standard likelihood-penalized model-selection
  criteria for choosing `n_components` (or `covariance_type`):
  $$\mathrm{BIC} = -2\,n\cdot\mathrm{score}(X) + p\log n, \qquad
    \mathrm{AIC} = -2\,n\cdot\mathrm{score}(X) + 2p$$
  where $p$ is the number of free parameters — $K{-}1$ for the weights
  (the simplex constraint removes one degree of freedom), $Kd$ for the
  means, plus the `covariance_type`-dependent count from §9 ($Kd(d{+}1)/2$
  full, $d(d{+}1)/2$ tied, $Kd$ diag, $K$ spherical). Lower is better for
  both; BIC penalizes $K$ more heavily for $n>7$ (favoring simpler
  models as data grows), AIC less so.

## 11. Scope

- **`covariance_type ∈ {"full","tied","diag","spherical"}`** — full
  scikit-learn parity on this axis; §9 is exactly why this project chose
  to implement all four rather than "full" alone: each is a distinct,
  short, worth-showing derivation, and the whole point of a *Gaussian
  mixture* (as opposed to KMeans) is the covariance structure.
- **`init_params ∈ {"kmeans","random"}`** — scikit-learn's other two
  options (`"k-means++"` seeding without a full KMeans run, and
  `"random_from_data"`, which draws initial means directly from training
  rows rather than initializing via responsibilities) are deferred; they
  are minor variations on the same idea and add little pedagogically once
  `"kmeans"` and `"random"` are both implemented.
- **No explicit-array initialization** (no `weights_init`/`means_init`/
  `precisions_init` constructor arguments) — unlike `KMeans`'s explicit
  `init` ndarray, this project's `GaussianMixture` has no fully
  deterministic, RNG-free path exposed publicly, so (§13) its reference
  tests compare the module-level EM step functions directly against
  scikit-learn's equivalent *given the same starting parameters* (which
  scikit-learn's constructor *does* accept), rather than comparing two
  full independently-initialized `fit()` calls.
- **No `warm_start`** — every `fit` call starts fresh.
- **No `n_features_in_`-mismatch tolerant covariance type validation
  beyond what `fit` already checks** — same posture as every other
  estimator here.

## 12. Algorithm (pseudocode)

```
GaussianMixture(n_components=1, covariance_type="full", tol=1e-3,
                reg_covar=1e-6, max_iter=100, n_init=1,
                init_params="kmeans", random_state=None)

fit(X):
    validate n_components >= 1, max_iter >= 1, tol >= 0, reg_covar >= 0,
             n_init >= 1, covariance_type in {full,tied,diag,spherical},
             init_params in {kmeans,random}
    X = check_array(X); require n_samples >= n_components
    rng = check_random_state(random_state)

    best = None
    for _ in range(n_init):
        resp0 = _init_kmeans_resp(X, K, rng)   or   _init_random_resp(n, K, rng)
        weights, means, covariances = _m_step(X, resp0, covariance_type, reg_covar)

        lower_bound = -inf
        for n_iter in range(1, max_iter + 1):
            log_prob_norm, log_resp = _e_step(X, weights, means, covariances, covariance_type)
            new_lower_bound = mean(log_prob_norm)
            weights, means, covariances = _m_step(X, exp(log_resp), covariance_type, reg_covar)
            if new_lower_bound - lower_bound < tol:
                lower_bound = new_lower_bound
                converged = True
                break
            lower_bound = new_lower_bound
        else:
            converged = False

        if best is None or lower_bound > best.lower_bound:
            best = (weights, means, covariances, lower_bound, n_iter, converged)

    self.weights_, self.means_, self.covariances_, self.lower_bound_, \
        self.n_iter_, self.converged_ = best
    _, log_resp = _e_step(X, *best[:3], covariance_type)   # re-sync to the KEPT run
    self.labels_ = argmax(log_resp, axis=1)
    return self

_e_step(X, weights, means, covariances, covariance_type):
    log_prob = _estimate_log_gaussian_prob(X, means, covariances, covariance_type)  # (n, K)
    weighted_log_prob = log_prob + log(weights)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)         # (n,)  = log p(x_i)
    log_resp = weighted_log_prob - log_prob_norm[:, None]
    return log_prob_norm, log_resp

_m_step(X, resp, covariance_type, reg_covar):
    Nk = resp.sum(axis=0) + 10*eps
    means = (resp.T @ X) / Nk[:, None]
    covariances = _estimate_covariances[covariance_type](X, resp, means, Nk, reg_covar)
    weights = Nk / n
    return weights, means, covariances

predict(X):         return argmax(_e_step(X, ...)[1], axis=1)
predict_proba(X):    return exp(_e_step(X, ...)[1])
score_samples(X):    return _e_step(X, ...)[0]
score(X):            return mean(score_samples(X))
sample(n_samples):   counts ~ Multinomial(n_samples, weights_)
                     per component k: mean_k + Cholesky(cov_k) @ standard_normal
bic(X): -2 * n * score(X) + n_parameters * log(n)
aic(X): -2 * n * score(X) + 2 * n_parameters
```

`_e_step`, `_m_step`, one `_estimate_covariances_*` per `covariance_type`,
`_estimate_log_gaussian_prob` (dispatch), `_init_kmeans_resp`,
`_init_random_resp`, and `_n_parameters` are factored out as module-level
functions, each independently unit-tested — mirroring KMeans's
`_assign`/`_update_centroids`/`_inertia` split.

## 13. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-laid, well-separated 1-D/2-D data with a fixed responsibility matrix: `_m_step` reproduces hand-computed `weights_`/`means_`/`covariances_` for each `covariance_type`. A single-component fit recovers the closed-form full-data mean/covariance exactly (no latent structure left to estimate). |
| Cross-check between covariance types (the objective-based analogue of a gradient check) | On data with a genuinely diagonal true covariance, `covariance_type="full"` recovers near-zero off-diagonal entries and its per-component log-likelihood matches `"diag"`'s to a tight tolerance — the same model expressed two ways must agree where `"diag"` is not actually restrictive. |
| Monotonicity | `lower_bound_` (avg. log-likelihood) is non-decreasing every EM iteration, checked by instrumenting a run — the direct consequence of §4. |
| KMeans limit | With `covariance_type="spherical"` and a tiny, fixed variance floor via `reg_covar`, `predict` on well-separated blobs agrees with `KMeans.fit_predict` on the same data up to a label permutation — §5 made concrete. |
| Independent reference | A tiny hand-built 2-point, 1-component case: closed-form mean/variance vs. `_m_step` output. |
| `n_init` | Best-of-`n_init` `lower_bound_` is `>=` any individual run's (a max over independently completed runs, the log-likelihood-flavored mirror of KMeans's inertia-min check). |
| Convergence | `n_iter_ <= max_iter`; `converged_` is `False` when a deliberately tiny `max_iter` cuts a run off; `True` on an easy, well-separated dataset with ample `max_iter`. |
| Determinism | Same `random_state` + data → identical `weights_`/`means_`/`covariances_`/`labels_`. |
| Contract | `predict`/`predict_proba`/`transform`-equivalents before `fit` raise `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; feature-count mismatch raises; `n_components < 1` raises; `n_components > n_samples` raises; bad `covariance_type`/`init_params` string raises; `repr` round-trips. |
| Behavioral | `make_blobs` with well-separated, near-isotropic centers: `labels_` recovers the true partition up to a label permutation with high agreement (KMeans-style permutation-matching check). |
| Generative | `sample(n_samples)` on a fitted 1-component model: the empirical mean/covariance of a large sample converges to `means_[0]`/`covariances_[0]`. `sample` output component labels appear with roughly `weights_`-proportional frequency for a multi-component fit. |
| Edge | `n_components == 1` (recovers the global mean/covariance exactly, no EM iteration needed to converge — `lower_bound_` improves by `~0` on iteration 2); `n_components == n_samples` (each point can become its own component); a near-singular covariance without `reg_covar` raises a clear `ValueError` rather than a raw `LinAlgError` or silent `nan`. |
| `bic`/`aic` | Matches a hand-computed value from `score(X)` and `_n_parameters()` for a small fitted model, for all four `covariance_type`s. |
| Reference (`-m reference`) | **Exact parity, given the same starting parameters**: seed both this project's `_m_step`/`_em` and `sklearn.mixture.GaussianMixture(weights_init=..., means_init=..., precisions_init=..., n_init=1, max_iter=k, tol=0)` from an identical fixed starting point, for each `covariance_type` — `weights_`, `means_`, `covariances_`, `lower_bound_` match after a fixed number of EM iterations (this isolates the E/M formulas from initialization randomness, the GaussianMixture analogue of KMeans's explicit-`init`-array exact-parity test, since this project's own public `init_params` has no RNG-free option). With the default `init_params="kmeans"` path (real RNG divergence from sklearn), held-out average log-likelihood is tolerance-only (`< 5%` relative, several seeds) — mirroring KMeans's `"k-means++"`/`"random"` tolerance tests. |

Plus `examples/gaussian_mixture.py` — seeded, runnable: fit `make_blobs`
with each `covariance_type`, report `converged_`/`lower_bound_`/`bic`,
compare hard `predict` labels to `KMeans` on the same data, and draw a
`sample()` from the fitted mixture.

## References

- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977), "Maximum Likelihood
  from Incomplete Data via the EM Algorithm", *Journal of the Royal
  Statistical Society, Series B* 39(1).
- Bishop, C.M., *Pattern Recognition and Machine Learning* (2006), §9.2–9.4
  — the E/M derivation, the KMeans limit, and the covariance-structure
  discussion this doc follows most closely.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §8.5, §14.3.7 (the K-means/EM connection).
- McLachlan, G., Peel, D., *Finite Mixture Models* (2000) — `reg_covar`-style
  covariance regularization and degenerate-solution avoidance.
