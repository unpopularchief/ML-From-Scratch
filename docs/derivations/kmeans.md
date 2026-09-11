# K-Means clustering

The finalized walkthrough for `scratchgrad.cluster.KMeans`, written per
`plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | data to cluster — unsupervised, no $y$ |
| $K$ | scalar $\ge 1$ | number of clusters (`n_clusters`) |
| $c_i$ | scalar $\in \{1, \dots, K\}$ | cluster assignment of point $i$ (`labels_`) |
| $\mu_k$ | $(d,)$ | centroid of cluster $k$ (`cluster_centers_`) |
| $\mathcal{C}_k$ | — | $\{i : c_i = k\}$, the points currently assigned to cluster $k$ |

## 2. The objective

K-means minimises the **within-cluster sum of squares** (WCSS, also called
inertia):

$$J(\{\mu_k\}, \{c_i\}) = \sum_{i=1}^n \lVert x_i - \mu_{c_i} \rVert^2
  = \sum_{k=1}^K \sum_{i \in \mathcal{C}_k} \lVert x_i - \mu_k \rVert^2$$

This is a joint minimisation over a **discrete** variable (the partition
$\{c_i\}$) and a **continuous** one (the centroids $\{\mu_k\}$). Jointly
optimal K-means is NP-hard, even for $K = 2$ in general dimension (Aloise
et al., 2009) — so, like `DecisionTree`'s greedy splitting, the practical
algorithm (Lloyd, 1957/1982) is a heuristic with a guarantee only on
*local* optimality: **block coordinate descent**, alternating exact
minimisation over one block with the other held fixed.

## 3. Lloyd's algorithm — two exact argmins

**Step 1 — assignment (fix $\mu$, minimise over $c$).** With the centroids
fixed, $J$ decomposes into an independent term per point:

$$J = \sum_i \lVert x_i - \mu_{c_i}\rVert^2$$

so the minimising assignment for point $i$ is simply the nearest centroid:

$$c_i \leftarrow \arg\min_k \lVert x_i - \mu_k \rVert^2$$

This is an exact argmin, computed once per point independently — no
iteration needed within the step. Implemented as the full $(n, K)$
distance matrix via `metrics.pairwise.euclidean_distance`, then a
row-wise `argmin`.

**Step 2 — update (fix $c$, minimise over $\mu$).** With the assignment
fixed, $J$ decomposes into an independent term per cluster:

$$J_k(\mu_k) = \sum_{i \in \mathcal{C}_k} \lVert x_i - \mu_k \rVert^2$$

Differentiating and setting to zero:

$$\frac{\partial J_k}{\partial \mu_k} = -2 \sum_{i \in \mathcal{C}_k} (x_i - \mu_k) = 0
  \quad\Longrightarrow\quad
  \mu_k \leftarrow \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} x_i$$

the cluster mean — the unique minimiser, since $J_k$ is a sum of convex
quadratics in $\mu_k$. This is the one calculus step in the whole
derivation, and it is also *why* the algorithm is married to squared
**Euclidean** distance specifically: the mean is the SSE-minimising
constant only under the L2 norm (§6 has the scope note on this).

## 4. Monotone decrease and termination

Both steps are the *exact* minimiser of their subproblem with the other
block frozen, so alternating them can never increase $J$:

$$J(\mu^{(t)}, c^{(t)}) \ge J(\mu^{(t)}, c^{(t+1)}) \ge J(\mu^{(t+1)}, c^{(t+1)})$$

(the first inequality is the assignment step re-minimising over $c$ with
$\mu^{(t)}$ fixed; the second is the update step re-minimising over $\mu$
with $c^{(t+1)}$ fixed). There are only finitely many ways to partition
$n$ points into $K$ labelled groups, and $J$ strictly decreases whenever
the partition actually changes (ties aside), so the sequence of distinct
partitions visited cannot repeat and the algorithm reaches a **fixed
point** — an assignment/centroid pair where neither step can improve
further — in finitely many iterations. This is a **local** optimum only:
which one depends entirely on where the centroids started, hence §5.

## 5. Initialisation — random vs. k-means++

**Random init** (`init="random"`): pick $K$ distinct training points
uniformly at random as the starting centroids. Simple, but a bad draw
(two seeds landing in the same true cluster) reliably produces a poor
local optimum.

**k-means++** (`init="k-means++"`, Arthur & Vassilvitskii, 2007 — the
default here and in scikit-learn since 0.24): seed centroids one at a
time, biased toward spreading them out.

1. Pick the first centroid uniformly at random from $X$.
2. For each remaining point $x$, let $D(x)$ be its squared distance to the
   *nearest centroid chosen so far*.
3. Pick the next centroid from $X$ with probability $\propto D(x)^2$ —
   points far from every existing centroid are far more likely to be
   picked next.
4. Repeat until $K$ centroids are chosen.

Intuition: a point already close to a chosen centroid is unlikely to be
picked again (it wouldn't add new coverage), while an unrepresented
region of the data is likely to get its own seed. Arthur & Vassilvitskii
prove this gives $\mathbb{E}[J] \le 8(\ln K + 2) \cdot J^*$ where $J^*$ is
the *global* optimum — an $O(\log K)$-competitive guarantee with no extra
iterations, which is why it dominates random init in practice and is the
default.

**Explicit array** (`init` = an `(K, d)` ndarray of starting centroids):
skips both of the above — useful for reproducing a specific run
deterministically, and for testing (§7).

**`n_init` restarts.** A single Lloyd run from one seed can still land in
a mediocre local optimum. `n_init` independent restarts (each with its
own k-means++/random draw, or the one explicit array run once) are fit
completely separately; the run with the lowest final `inertia_` is kept.
This — not `tol`/`max_iter` — is the real defence against bad local
optima.

## 6. Convergence and stopping

Two stopping conditions per run, whichever comes first:

- `max_iter` iterations reached.
- **Centroid shift below `tol`.** Comparing raw centroid movement is
  scale-dependent (the same absolute shift means "converged" on a
  millimetre-scale feature and "still moving" on a kilometre-scale one),
  so — matching scikit-learn — `tol` is scaled by the data's mean
  per-feature variance: stop when
  $$\sum_k \lVert \mu_k^{(t+1)} - \mu_k^{(t)} \rVert^2 \;\le\; \texttt{tol} \cdot \frac{1}{d}\sum_{j=1}^d \mathrm{Var}(X_{:,j})$$
  computed once from the input data at the start of `fit`.

`n_iter_` records the iteration count of the **kept** (lowest-inertia)
run.

## 7. Scope

- **Euclidean distance only** — no `metric` parameter. §3's update-step
  derivation is specific to squared L2 (the mean minimises squared L2
  error); an arbitrary metric would need a different, generally
  non-closed-form "medoid" update (that is
  [k-medoids](https://en.wikipedia.org/wiki/K-medoids)/PAM, out of scope).
- **Hard assignment only** — no fuzzy/soft K-means. `GaussianMixture`
  (later in M2) is the soft-assignment generalisation via EM.
- **No `algorithm` parameter** — no Elkan's triangle-inequality
  acceleration (`plan.md` §6 spirit: an acceleration structure, not part
  of the method's statistics, mirrors KNN having no `algorithm` param).
- **Empty clusters.** An assignment step can leave a cluster with zero
  points (more likely with `init="random"` or a bad k-means++ draw, or
  more centroids than distinct points). Matching scikit-learn: the empty
  cluster's centroid is relocated to whichever *currently assigned* point
  is farthest from its own cluster's centroid — that point becomes a
  singleton cluster, "stealing" the worst-fit outlier from an
  overcrowded cluster. Without this, the run would either crash on
  `mean([])` or silently run with fewer than `K` effective clusters.

## 8. Algorithm (pseudocode)

```
KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4,
       random_state=None)

fit(X):
    validate n_clusters >= 1, max_iter >= 1, tol >= 0
    X = check_array(X); require n_samples >= n_clusters
    tol_scaled = tol * mean(var(X, axis=0))
    rng = check_random_state(random_state)

    best = None
    for _ in range(n_init):                      # n_init=1 when init is an explicit array
        centers = _init_centroids(X, init, n_clusters, rng)
        for it in range(1, max_iter + 1):
            dist = euclidean_distance(X, centers)  # (n, K)
            labels = argmin(dist, axis=1)
            new_centers = _update_centroids(X, labels, centers, n_clusters)  # handles empties
            shift = sum(||new_centers - centers||^2)
            centers = new_centers
            if shift <= tol_scaled:
                break
        inertia = sum(min(dist, axis=1))           # recomputed against the final centers
        if best is None or inertia < best.inertia:
            best = (centers, labels, inertia, it)

    self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
    return self

_init_centroids(X, init, K, rng):
    if init is an ndarray: return it.copy()                     # 1 run, no randomness
    if init == "random":   return X[rng.choice(n, K, replace=False)]
    if init == "k-means++":
        centers = [X[rng.integers(n)]]
        for _ in range(K - 1):
            D2 = min_k(euclidean_distance(X, centers)) ** 2      # dist to nearest chosen center
            centers.append(X[rng.choice(n, p=D2 / D2.sum())])
        return stack(centers)

_update_centroids(X, labels, old_centers, K):
    centers = old_centers.copy()
    for k in range(K):
        members = X[labels == k]
        if len(members) == 0:
            far_point = argmax(||X - old_centers[labels]||^2)     # farthest point from ITS OWN centroid
            centers[k] = X[far_point]
            labels[far_point] = k                                 # steal it into the empty cluster
        else:
            centers[k] = members.mean(axis=0)
    return centers

predict(X):    return argmin(euclidean_distance(X, self.cluster_centers_), axis=1)
transform(X):  return euclidean_distance(X, self.cluster_centers_)      # (n, K)
score(X):      return -sum(min(euclidean_distance(X, self.cluster_centers_), axis=1) ** 2)
```

`_assign`, `_update_centroids`, `_init_random`, `_init_kmeanspp`, and
`_inertia` are factored out as module-level functions so each piece is
unit-tested directly, mirroring the `_*` helpers in `tree/` and
`ensemble/`.

## 9. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-laid 1-D/2-D data with obviously-separated clusters and an explicit `init` array: `_assign`/`_update_centroids` produce the exact expected labels/centroids by hand computation. |
| Monotonicity (the objective-based analogue of a gradient check) | `inertia_` after each Lloyd iteration is non-increasing, checked by instrumenting a run on random data — the direct consequence of §4's two-exact-argmins argument. |
| Independent reference | On a tiny dataset (`n` small enough, e.g. $n=8, K=2$), brute-force the WCSS-minimising partition over *all* $2^n/2$ assignments and confirm k-means++ with enough `n_init` restarts finds it (or ties it). |
| k-means++ vs. random | On a constructed case with one dense cluster and one sparse outlier region, `n_init=1` k-means++ finds the right partition far more reliably across seeds than `n_init=1` random init (measured over many seeds, not a single-seed assertion — the RandomForest/AdaBoost lesson). |
| Empty cluster | Construct an input plus explicit `init` guaranteed to empty a cluster on iteration 1; confirm no crash, the singleton-steal behaviour, and `labels_` still uses all `K` labels. |
| `n_init` | Best-of-`n_init` inertia is `<=` any individual run's inertia (it's a min over independently completed runs). |
| Convergence | `n_iter_ <= max_iter`; a tiny `max_iter` (e.g. 1) still returns a valid (if unconverged) result, no exception. |
| Determinism | Same `random_state` (or `init` array) + same data → identical `cluster_centers_` / `labels_`. |
| Contract | `predict`/`transform` before `fit` raise `NotFittedError`; `fit` returns `self`; hyperparameters unchanged after `fit`; feature-count mismatch in `predict`/`transform` raises; `n_clusters < 1` raises; `n_clusters > n_samples` raises; bad `init` string/shape raises; `repr` round-trips. |
| Behavioral | `make_blobs` with well-separated centers: `labels_` recovers the true partition up to a label permutation (checked by brute-force permutation matching over the small `K`, the KNN-style independent check) with high agreement. |
| Edge | single feature; `n_clusters == n_samples` (`inertia_` == 0, one point per cluster); `n_clusters == 1` (single centroid == the global mean, `inertia_` == total variance × n). |
| Reference (`-m reference`) | With an explicit `init` array (no RNG involved on either side), Lloyd's iteration matches `sklearn.cluster.KMeans(init=..., n_init=1, algorithm="lloyd")` **exactly** — same `cluster_centers_`, same `labels_`, same `inertia_`, same `n_iter_` (this is the deterministic-path exact-parity test, the KMeans analogue of the tree's `max_features=None` path). With `init="k-means++"`/`"random"`, our `Generator` stream diverges from sklearn's, so those are tolerance-only: comparable `inertia_` (`< 5%` relative) over several seeds. |

Plus `examples/kmeans.py` — seeded, runnable: cluster `make_blobs` and
report inertia; an elbow-method sweep over `n_clusters`; a k-means++ vs.
random init comparison over several seeds.

## References

- Lloyd, S. (1982, written 1957), "Least squares quantization in PCM",
  *IEEE Transactions on Information Theory* 28(2).
- MacQueen, J. (1967), "Some methods for classification and analysis of
  multivariate observations", *Proc. 5th Berkeley Symposium*.
- Arthur, D. & Vassilvitskii, S. (2007), "k-means++: The Advantages of
  Careful Seeding", *SODA*.
- Aloise, D., Deshpande, A., Hansen, P., Popat, P. (2009), "NP-hardness of
  Euclidean sum-of-squares clustering", *Machine Learning* 75(2).
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*,
  2nd ed., §14.3.6.
