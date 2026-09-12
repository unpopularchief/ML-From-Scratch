# DBSCAN

The finalized walkthrough for `scratchgrad.cluster.DBSCAN`, written per
`plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

## 1. Problem setup and notation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(n, d)$ | data to cluster — unsupervised, no $y$ |
| $\varepsilon$ (`eps`) | scalar $> 0$ | neighborhood radius |
| `min_samples` | scalar $\ge 1$ | minimum neighbor count (including the point itself) to be a **core point** |
| $N_\varepsilon(x)$ | — | $\{x' \in X : d(x, x') \le \varepsilon\}$, the $\varepsilon$-neighborhood of $x$ |
| core point | — | $x$ is core iff $\lvert N_\varepsilon(x) \rvert \ge$ `min_samples` |
| border point | — | non-core point lying in some core point's $N_\varepsilon$ |
| noise point | — | neither core nor border — `labels_ == -1` |

Unlike every algorithm so far, **there is no calculus here.** K-means
minimises within-cluster SSE; every supervised method minimises some loss.
DBSCAN (Ester, Kriegel, Sander, Xu, 1996) instead *defines* a cluster
combinatorially, via a reachability relation on the $\varepsilon$-neighbor
graph. The "derivation" is a correctness argument for why single-pass graph
expansion recovers exactly the sets that definition describes — the
DBSCAN analogue of DecisionTree's "no gradient, an NP-hardness argument for
why greedy is used" tier.

## 2. The definition: density-reachability, not an objective

- $y$ is **directly density-reachable** from $x$ if $x$ is a core point and
  $y \in N_\varepsilon(x)$.
- $y$ is **density-reachable** from $x$ if there is a chain $x = p_0, p_1,
  \dots, p_k = y$ where each $p_{i+1}$ is directly density-reachable from
  $p_i$ (every $p_0, \dots, p_{k-1}$ must therefore be core — only a core
  point's neighborhood propagates reachability).
- $x$ and $y$ are **density-connected** if some core point $z$
  density-reaches both.
- A **cluster** $C$ is a maximal, non-empty subset of $X$ such that every
  pair in $C$ is density-connected, and every point density-reachable from
  a point in $C$ is also in $C$.

**Key lemma (Ester et al., Lemma 2):** a cluster is uniquely determined by
*any one* of its core points — i.e. if $p$ is core and $C$ is the cluster
containing $p$, then $C$ equals exactly the set of points density-reachable
from $p$. This is what licenses the algorithm in §3: expand outward from
one core point at a time and every point that expansion reaches belongs to
the same cluster, full stop, with no need to ever revisit or merge clusters
after the fact.

**What this lemma does *not* cover — border points.** The lemma is about
core points only. A border point $b$ that sits in $N_\varepsilon$ of two
core points from *different* clusters is density-reachable from both, so
the definition above doesn't pick a unique cluster for it — the original
paper acknowledges this and leaves it to the implementation. §4 fixes it by
matching scikit-learn's rule exactly, rather than inventing a "principled"
tie-break with no basis in the paper.

## 3. Algorithm: single-pass expansion

Because of the lemma, clustering reduces to: scan for an unvisited core
point, expand the full set of points density-reachable from it (that's one
whole cluster, by the lemma), mark them all visited, repeat. Points touched
by no expansion are noise.

```
fit(X):
    validate eps > 0, min_samples >= 1
    X = check_array(X)
    dist = pairwise_distance(X, X)                  # (n, n)
    neighbor_mask = dist <= eps                      # includes the diagonal (d=0)
    neighbor_counts = sum(neighbor_mask, axis=1)
    is_core = neighbor_counts >= min_samples
    neighborhoods = [where(neighbor_mask[i])[0] for i in range(n)]   # ascending index

    labels = full(n, -1)                             # -1 = noise, until claimed
    next_label = 0
    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue                                 # already claimed, or can't start a cluster
        i = seed
        stack = []
        while True:
            if labels[i] == -1:
                labels[i] = next_label
                if is_core[i]:
                    for neighbor in neighborhoods[i]:
                        if labels[neighbor] == -1:
                            stack.append(neighbor)     # queue it for expansion
            if not stack:
                break
            i = stack.pop()                            # LIFO -> depth-first
        next_label += 1

    self.labels_ = labels
    self.core_sample_indices_ = where(is_core)[0]
    self.components_ = X[self.core_sample_indices_]
    return self
```

No iteration count, no convergence tolerance, no initialisation, no `n_init`
— one deterministic pass over a graph that's fully determined by `eps` and
`min_samples`. `min_samples` counts the point itself (matches the original
paper and scikit-learn), so `min_samples=1` makes every point core.

## 4. The border-point tie, resolved by matching scikit-learn exactly

§2 flagged that a border point touching two clusters' core points has no
principled single answer. The rule above resolves it as **"whichever core
point's expansion reaches it first, in this exact traversal order"** — a
LIFO stack (depth-first), scanning seed candidates in ascending row index
and each core point's own neighbor list in ascending row index. This isn't
an arbitrary choice made from scratch: it is scikit-learn's own
`dbscan_inner` traversal (a depth-first search with an explicit stack, not
a queue), reproduced exactly, including neighbor iteration order — see
`docs/derivations/dbscan.md`'s companion reference tests. This is the
*same policy decision* KMeans made with the empty-cluster relocation rule
(§7 of `kmeans.md`): where the definition is genuinely ambiguous, match the
reference implementation rather than inventing new semantics.

The payoff: because DBSCAN has **no RNG anywhere** in its own definition
(unlike KMeans's k-means++/random paths, or every ensemble's bootstrap /
feature subsampling), matching this traversal gives **exact
`labels_`/`core_sample_indices_` parity with `sklearn.cluster.DBSCAN` on
*every* input**, not just a deterministic special case — verified
empirically across dozens of seeds, cluster shapes, and both supported
metrics (§7).

## 5. Why no `predict`

A K-means/GaussianMixture cluster is a region of space — nearest centroid,
or highest density under a fitted distribution — so a new point trivially
gets assigned. A DBSCAN cluster is a property of the *realized*
$\varepsilon$-neighbor graph of the training set: whether a new point is
core, border, or noise (and which cluster it would join) can depend on
exactly which existing points happen to be its neighbors, and inserting it
could in principle even change an existing point's core/border status.
There is no way to answer "which cluster does this new point belong to"
without effectively re-running the graph construction. This is why
`sklearn.cluster.DBSCAN` itself has no `predict` — only `fit`/`fit_predict`
— and why this implementation matches that rather than inventing an
approximate nearest-core-point heuristic with no basis in the algorithm's
own definition.

## 6. Scope

- **`metric ∈ {"euclidean", "manhattan"}`**, mirroring `KNeighborsClassifier`
  (`docs/derivations/knn.md`) — both already exist in
  `scratchgrad.metrics.pairwise`, and unlike K-means's update step (§7 of
  `kmeans.md`), nothing about density-reachability depends on squared L2
  specifically; any metric with a well-defined ball works.
- **Brute force only, no `algorithm` parameter** — no ball-tree/kd-tree
  acceleration, the same stance as `KNeighborsClassifier` and `KMeans`: an
  acceleration structure is not part of the method's statistics.
  $O(n^2 d)$ time and $O(n^2)$ memory for the full distance matrix (this
  implementation, like `KMeans`/`KNeighborsClassifier`, is not built for
  large $n$).
- **No `predict`** — `fit`, `fit_predict` only (§5).
- **No `sample_weight`** in this PR — scikit-learn's `DBSCAN` lets a
  point's weight count toward its neighbors' core-point threshold; deferred
  as a follow-up, matching how `DecisionTreeClassifier`'s `sample_weight`
  was its own PR (#10) after the base tree shipped.
- **No `precomputed` metric / sparse input** — out of scope for the same
  reason no estimator here accepts a precomputed distance matrix elsewhere.

## 7. What the tests check

| Tier | Check |
| --- | --- |
| Analytic | Hand-laid 1-D/2-D data: a chain of points each within `eps` of the next forms one cluster; an isolated point below `min_samples` neighbors is noise; a hand-verified core/border/noise labelling on a small constructed example (a "bridge" point that is border to one cluster). |
| Independent reference | An independent, unvectorised triple-loop implementation of §2's definitions directly (build the $\varepsilon$-graph with `for`-loops, compute core points, then BFS/DFS by hand in a second, differently-structured pass) agrees with the fitted `labels_` up to the label-permutation freedom clusters have — the DecisionTree/KMeans-style "no gradient, an independent recomputation stands in" tier. |
| Correctness invariants | Every core point is in a cluster (never noise); every point in `neighborhoods[i]` of a core point `i` is core, border, or noise-then-claimed (never left as `-1` once `i` is processed); `is_core` count matches `len(core_sample_indices_)`; every cluster contains at least one core point (falls directly out of the cluster definition in §2). |
| Metric | Same data, same `eps`/`min_samples`, `metric="manhattan"` vs `"euclidean"` can disagree (different balls) — checked on a constructed anisotropic example where they're known to disagree, not asserted equal. |
| Contract | `fit` returns `self`; hyperparameters unchanged after `fit`; `eps <= 0` raises; `min_samples < 1` raises; unknown `metric` raises; `repr` round-trips; no `predict` method exists (`AttributeError` — documented as intentional, not a bug). |
| Behavioral | `make_moons` (two interleaving, non-convex arcs): DBSCAN recovers the two arcs as two clusters (checked by best-permutation label agreement, matching KMeans's own behavioral test style) where **K-means provably cannot** — a small side-by-side comparison making DBSCAN's actual selling point (arbitrary cluster shape, no `n_clusters`) concrete rather than asserted. A dataset with obvious outliers: outliers land in `labels_ == -1`. |
| Edge | All points identical → one cluster if `min_samples <= n`, all core; every point pairwise farther than `eps` apart → every point noise (`min_samples > 1`) or every point its own singleton cluster (`min_samples == 1`); single feature; `min_samples=1` (every point is core by definition). |
| Reference (`-m reference`) | **Exact parity** with `sklearn.cluster.DBSCAN(algorithm="brute")` on `labels_` **and** `core_sample_indices_` — not a tolerance test — verified across many seeds, cluster counts, both metrics, and deliberately eps-tuned "double-border-point" cases designed to exercise the tie in §4. This is a *stronger* parity result than any prior algorithm here gets by default: no RNG exists on either side to diverge, so there is no fallback tolerance test needed at all. |

Plus `examples/dbscan.py` — seeded, runnable: cluster `make_moons` (the
non-convex case KMeans cannot solve), a `make_blobs` case with injected
noise points, and an `eps` sensitivity sweep showing how too-small an `eps`
fragments a blob into noise and too-large merges separate blobs.

## References

- Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996), "A Density-Based
  Algorithm for Discovering Clusters in Large Spatial Databases with
  Noise", *KDD-96*.
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., Xu, X. (2017),
  "DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN",
  *ACM TODS* 42(3).
- `scikit-learn`'s `sklearn/cluster/_dbscan.py` and
  `_dbscan_inner.pyx` — the traversal §4 matches exactly.
