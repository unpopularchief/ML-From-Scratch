# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `scratchgrad.nn.Dropout`/`BatchNorm1d`, and `Module`'s `training`/`eval`
  flag — the third M3 unit. `Module.training` (class attribute, default
  `True`) plus `train()`/`eval()` methods, read only by these two new
  layers -- `Linear`/activations/losses are unaffected. `Dropout(p,
  random_state)`: inverted dropout (`bernoulli(1-p)/(1-p)` mask in
  training, identity with no RNG draw in eval), so `E[y]=x` in training and
  no eval-time rescale is needed (matches `torch.nn.Dropout`).
  `BatchNorm1d(num_features, eps, momentum)`: per-feature normalization
  over the batch in training (with a Bessel-corrected running mean/var
  update, exactly matching `torch.nn.BatchNorm1d` even though the batch
  itself is normalized with the biased variance) or over the running
  stats in eval -- each mode has its own, algebraically distinct backward
  formula (eval's has no cross-sample coupling since the running stats
  aren't functions of the current batch), both independently
  gradient-checked. `tests/reference/test_dropout_batchnorm.py` confirms
  **exact** `BatchNorm1d` parity against `torch.nn.BatchNorm1d` in both
  modes, and exact `Dropout` eval-mode parity (training mode is RNG-stream
  dependent, checked statistically instead). `examples/nn.py` extended to
  `Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear ->
  BCEWithLogitsLoss`, switching every stateful layer to `.eval()` before
  each accuracy/loss checkpoint and back to `.train()` afterward.

- `scratchgrad.nn` — hand-derived neural network building blocks, the
  second M3 unit: `Module` (the shared `forward`/`backward`/
  `parameters()`/`grads()` interface -- `parameters()`/`grads()` return
  same-order `list[ndarray]`s directly consumable by
  `optim.step(params, grads)`), `Linear` (`Y = XW + b`, backward via the
  matmul chain rule), activations `ReLU`/`Sigmoid`/`Tanh`/`Softmax` (each
  with its own backward; `Softmax`'s is the full per-row
  Jacobian-vector product `diag(y) - y y^T`, not just a forward-only
  helper), losses `MSELoss`/`BCEWithLogitsLoss`/`CrossEntropyLoss` (the
  latter two take raw logits, fused with the sigmoid/softmax link
  internally for stability -- the same move as `LogisticRegression`'s
  `softplus(z) - yz` loss), and init schemes `zeros`/`xavier_uniform`/
  `he_normal`. No `Module` subclassing of `base.Estimator` (a layer has
  neither `fit` nor `predict`) and no shared `training`/`eval` flag yet
  (deferred to the `Dropout`/`BatchNorm` unit that actually needs it).
  `tests/reference/test_nn.py` confirms **exact** parity against
  `torch.nn.Linear`/activations/losses (same tier as `optim/`'s PyTorch
  parity, not a tolerance-only outcome comparison). `examples/nn.py`
  hand-wires `Linear -> ReLU -> Linear -> BCEWithLogitsLoss` and trains
  it with `optim.Adam` on `make_moons` (74.7% -> 99.3% accuracy) -- `nn`
  and `optim` working together end-to-end. New `nn/` package.

- `scratchgrad.optim` — first-order optimizers, the first M3 unit: `SGD`
  (plain `theta -= lr*grad`), `Momentum` (Polyak 1964 heavy-ball, a
  running velocity buffer), `Nesterov` (accelerated gradient, reformulated
  per Sutskever et al. 2013 to need only the gradient at the current point
  rather than a lookahead evaluation — the interface every optimizer here
  shares, `step(params: list[ndarray], grads: list[ndarray])` mutating in
  place, has no mechanism to evaluate a shifted point), `RMSprop` (Hinton
  2012, per-coordinate step scaled by a running average of squared
  gradients), and `Adam` (Kingma & Ba 2015, bias-corrected Momentum +
  RMSprop). All five operate on generic `list[ndarray]` parameters/
  gradients rather than one flat vector, so a future layer's differently-
  shaped `W`/`b` need no flattening; no shared `BaseOptimizer` class (only
  three lines of lazy state-buffer allocation are actually common across
  four of the five — not enough to justify one, `plan.md` §8 mistake 2).
  `torch` added to the `reference` extra (scikit-learn parity tests since
  M1; PyTorch joins at M3, per `plan.md` §7) — `tests/reference/
  test_optim.py` confirms **exact** parameter-trajectory parity against
  `torch.optim.SGD`/`RMSprop`/`Adam` (these are the same published update
  formulas, not a different algorithm converging to the same outcome,
  unlike every prior reference-test tier). New `optim/` package.

## [0.2.0] — 2026-09-17

### Added

- `scratchgrad.svm.LinearSVM` — soft-margin linear support vector
  classifier, fitted by minimising the primal hinge-loss objective
  directly (no kernel trick, no dual): `J(w,b) = ½‖w‖² + C·Σ max(0, 1 −
  yᵢ(w·xᵢ+b))`, matching `sklearn.svm.LinearSVC(loss="hinge")`'s
  documented primal exactly so `C` needs no rescaling. The hinge term is
  non-differentiable at `yᵢzᵢ = 1`; a consistent subgradient is picked
  there (the flat side of the kink, mirroring `Lasso`'s L1 treatment) and
  minimised by the subgradient method with a **diminishing step size**
  `η_t = lr/√t` and **best-iterate ("pocket") tracking** — both needed
  because, unlike smooth gradient descent, a fixed step size on a
  non-smooth objective need not converge and the objective is not
  guaranteed to decrease at every individual step. New `svm/` package.
  Binary only (multiclass deferred); no `predict_proba` (a hinge SVM has
  no native probabilistic output without a separate Platt-scaling
  calibrator, out of scope); no `random_state` — the algorithm has no
  randomness anywhere (deterministic zero initialisation), so `fit` is
  reproducible without one, like `DBSCAN`/`PCA`. **No exact scikit-learn
  parity** — `LinearSVC` solves the dual via liblinear's coordinate
  descent, a different algorithm from this primal subgradient method, so
  reference tests compare held-out accuracy and decision-boundary sign
  agreement within a tolerance, the same tier `RandomForest`/
  `GradientBoosting` use. This completes M2 (`v0.2.0`) — kernel SVM/SMO
  remains explicitly optional per `plan.md` and is not implemented.

- `scratchgrad.decomposition.PCA` — principal component analysis. The
  directions that maximise projected variance are shown (Lagrangian on
  `w^T C w` s.t. `||w||=1`) to be exactly the eigenvectors of the
  covariance matrix `C`, ordered by eigenvalue — and, via a Pythagorean
  decomposition of each point, that same set of directions simultaneously
  minimises squared reconstruction error, so "capture the most variance"
  and "reconstruct best" are the same problem. Computed via the SVD of
  the centered data (`X_c = U Σ V^T`) rather than an explicit
  `eigh(cov(X))` — squaring the condition number by forming `X_c^T X_c`
  is exactly the numerical-stability trap `eigh` would otherwise walk
  into — reading `components_`/`explained_variance_` straight off `Σ, V`
  instead. Eigenvector sign is fixed deterministically (largest-magnitude
  entry positive per component), matching scikit-learn's convention, so
  results are reproducible and comparable across runs. Per `plan.md` §6's
  explicit callout for this algorithm, a from-scratch
  `_power_iteration_pca` (deflation-based power iteration on the
  covariance matrix — an independent algorithm, not another library
  decomposition) is kept and tested to agree with the SVD path on both
  `components_` and eigenvalues — that agreement is itself the lesson,
  mirroring the `nn`/`autograd` duplication described in `plan.md` §1.
  Deliberately minimal relative to scikit-learn's `PCA`: `n_components`
  is `int | None` only (no float variance-ratio threshold, no `"mle"`),
  no `whiten`, no `svd_solver` choice — each is a convenience layered on
  the same eigenvalues, not new math, and can be added later if a
  concrete need shows up. **Exact parity with `sklearn.decomposition.PCA`**
  (deterministic on both sides — no RNG anywhere in a full SVD) on
  `components_`, `explained_variance_`, `explained_variance_ratio_`,
  `singular_values_`, and `transform`/`inverse_transform` output.

- `scratchgrad.cluster.GaussianMixture` — a Gaussian mixture model fitted
  by Expectation-Maximization (Dempster, Laird, Rubin, 1977). The
  soft-assignment, shape-and-size-learning generalisation of `KMeans`:
  where KMeans hard-assigns each point to its nearest centroid, EM
  computes a posterior **responsibility** for every component
  (`predict_proba`) and, in the KMeans-equivalent limit (shared, shrinking
  spherical covariance), the two algorithms coincide exactly (see the
  derivation §5). The observed-data log-likelihood is a log of a sum with
  no closed-form maximizer; EM instead alternates an E-step (posterior
  responsibilities via Bayes' rule) and an M-step (responsibility-weighted
  MLE of every parameter), provably non-decreasing the log-likelihood
  every iteration — never guaranteed to reach the global optimum, so
  `n_init` restarts matter exactly as they do for KMeans. All four
  `covariance_type`s (`"full"`, `"tied"`, `"diag"`, `"spherical"`), each
  its own closed-form weighted-MLE derivation, not an approximation of
  `"full"`. `init_params ∈ {"kmeans", "random"}` — `"kmeans"` (the
  default) runs this project's own `KMeans` once and one-hot encodes its
  labels, a genuine cross-module reuse. `reg_covar` guards against a
  singular covariance the same way `GaussianNB`'s `var_smoothing` guards
  a zero variance; a covariance that stays singular anyway raises a clear
  `ValueError` naming the fix rather than a raw `LinAlgError`. Beyond
  `fit`/`predict`: `score`/`score_samples` (the fitted log-likelihood),
  `sample` (draws real synthetic points — the concrete payoff of being a
  generative model, which KMeans's pure partition never was), and
  `bic`/`aic` (standard likelihood-penalized model-selection criteria for
  choosing `n_components`). **Exact parity with
  `sklearn.mixture.GaussianMixture`** given an identical starting point
  for all four `covariance_type`s (verified by seeding both sides'
  E/M-step math from the same fixed `weights`/`means`/`covariances` and
  comparing after a fixed number of iterations — this project's own
  `init_params` has no RNG-free path the way KMeans's explicit `init`
  array does); the default `init_params="kmeans"` path is tolerance-only
  (`< 5%` relative log-likelihood/BIC gap), like every other
  RNG-dependent estimator here. Derivation: `docs/derivations/gaussian_mixture.md`;
  exports in both `__init__.py`s; README + ROADMAP updated;
  `examples/gaussian_mixture.py`.

- `scratchgrad.cluster.DBSCAN` — density-based clustering. Unlike every
  prior algorithm here, there is no objective being minimised: a cluster
  is *defined* via density-reachability on the `eps`-neighbor graph
  (Ester, Kriegel, Sander, Xu, 1996) — a point is a **core point** if it
  has at least `min_samples` neighbors within `eps` (counting itself); a
  cluster is the full set of points reachable from one core point through
  a chain of core points; points reached by no expansion are **noise**
  (`labels_ == -1`). No `n_clusters` to choose. A border point touching
  two different clusters' core points has no unique answer under the
  definition itself — resolved by matching `sklearn.cluster.DBSCAN`'s own
  depth-first, stack-based traversal order exactly, rather than inventing
  a new tie-break. `metric ∈ {"euclidean", "manhattan"}`. **No RNG
  anywhere** in the algorithm (no `init`, no bootstrap), so this achieves
  **exact** `labels_`/`core_sample_indices_` parity with
  `sklearn.cluster.DBSCAN(algorithm="brute")` on every input tested, not
  just a deterministic special case — a stronger parity result than any
  prior algorithm gets by default. No `predict` — DBSCAN's clusters are a
  property of the training set's own realized neighbor graph, with no
  principled out-of-sample rule (matches scikit-learn's own `DBSCAN`,
  which has no `predict` either). Brute-force `(n, n)` distance matrix, no
  `algorithm` acceleration parameter, mirroring `KMeans`/
  `KNeighborsClassifier`. Derivation: `docs/derivations/dbscan.md`;
  example: `examples/dbscan.py`.
- `scratchgrad.cluster.KMeans` — the first unsupervised algorithm, and
  the first entry in the new `cluster/` package. Lloyd's algorithm:
  alternates a nearest-centroid assignment step with a per-cluster-mean
  update step, each the exact argmin of its subproblem holding the other
  fixed, so the within-cluster sum of squares (`inertia_`) is
  non-increasing every iteration and the run terminates at a local
  optimum. `init` is `"k-means++"` (D(x)²-weighted seeding, the default),
  `"random"` (uniform draw of distinct points), or an explicit
  `(n_clusters, n_features)` array (deterministic, forces `n_init=1`).
  `n_init` independent restarts keep the lowest-inertia run. An empty
  cluster (zero points assigned) has its centroid relocated to whichever
  point is farthest from its own cluster's centroid, excluded from that
  cluster's mean — matches `sklearn.cluster.KMeans`'s relocation rule
  exactly. `tol` is scaled by the data's mean per-feature variance.
  `predict`, `fit_predict`, `transform` (distance to every centroid),
  `fit_transform`, `score` (`-inertia_`, sklearn's convention). Euclidean
  distance only — no `metric` parameter, since the mean-minimises-SSE
  argument in the update step is specific to squared L2. With an explicit
  `init` array there is no RNG anywhere in Lloyd's algorithm, so this
  matches `sklearn.cluster.KMeans(algorithm="lloyd")` **exactly**
  (`cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, including the
  empty-cluster relocation path) — the KMeans analogue of the
  tree's `max_features=None` deterministic-parity path.
  `"k-means++"`/`"random"` draw from our own `Generator`, so those are
  tolerance-only (comparable inertia). Derivation:
  `docs/derivations/kmeans.md`; example: `examples/kmeans.py`.
- `scratchgrad.ensemble.GradientBoostingClassifier` — gradient boosting
  for classification: functional gradient descent on the log loss. Each
  round fits a `DecisionTreeRegressor` to the pseudo-residual (the
  negative loss gradient w.r.t. the raw score — `y − p` for the binomial
  deviance), then replaces the tree's leaf values with a one-step Newton
  estimate of the loss-minimising constant on each leaf ("TreeBoost");
  `learning_rate` shrinks the update. Multiclass fits one tree per class
  per round against the multinomial-deviance residuals (with the
  `(K−1)/K` leaf factor); `K = 2` uses the single-score binomial path.
  `subsample < 1` fits each tree on a fresh row subsample (stochastic
  gradient boosting), seeded by `random_state`. `init` is the weighted
  base-rate log-odds / log-priors (`"prior"`) or zero. `sample_weight`
  flows through the init score, the Newton leaves, the base fits, and
  `train_score_` (the per-round training deviance). `decision_function`,
  `predict_proba` (a real posterior, unlike AdaBoost's score calibration),
  `staged_predict` / `staged_predict_proba` /
  `staged_decision_function`, `feature_importances_`. Held-out accuracy
  and log loss track `sklearn.ensemble.GradientBoostingClassifier` to a
  tolerance (their `friedman_mse` splitter and RNG feature order differ
  from our `squared_error` base tree, so trees are not identical).
  Log-loss only — `loss="exponential"` (AdaBoost) and
  `GradientBoostingRegressor` are out of scope. The other half of the M2
  "supervised ensembles" track. Derivation:
  `docs/derivations/gradient_boosting.md`; example:
  `examples/gradient_boosting.py`.
- `scratchgrad.tree.DecisionTreeRegressor` — the regression CART tree.
  Splits maximise the within-node variance reduction
  `ΔH = H(S_t) − (N_L/N_t)·H(S_L) − (N_R/N_t)·H(S_R)` with `H` the weighted
  variance of `y` (the `"squared_error"` criterion), which by the law of
  total variance is the between-group variance of the split, i.e. the
  greedy drop in training sum-of-squared-error. Each leaf predicts the
  weighted mean of its targets (the SSE-minimising constant); `score` is
  `R²`. Shares the split search, `_Node`, `max_features` / `random_state`
  per-node subsampling, and the `sample_weight` hook with
  `DecisionTreeClassifier` — exact scikit-learn parity on the
  deterministic path while nodes stay large, held-out-`R²` tolerance once
  they get small enough for several features to tie bit-for-bit (or once
  `max_features` makes the RNG streams diverge).
  `absolute_error` / `friedman_mse` / `poisson` are out of scope.
  Prerequisite for `GradientBoosting` and a future `RandomForestRegressor`.
  Derivation: `docs/derivations/decision_tree_regressor.md`; example:
  `examples/decision_tree_regressor.py`.
- `scratchgrad.ensemble.AdaBoostClassifier` — AdaBoost (SAMME): forward
  stagewise fitting of a multi-class exponential loss with decision-tree
  weak learners (a depth-1 stump by default, `max_depth` exposed). Each
  round refits the tree on the rows reweighted toward the previous round's
  mistakes and adds it to the vote with weight
  `alpha_m = log((1 - err_m) / err_m) + log(K - 1)`; `learning_rate`
  shrinks the step. Stops early on a perfect round or a round that cannot
  beat random guessing (and raises if that is the first round).
  `decision_function`, `predict_proba` (scikit-learn's monotone score
  calibration), `staged_predict` / `staged_score`, `estimator_weights_` /
  `estimator_errors_`, `alpha`-weighted `feature_importances_`.
  Multiclass; `K = 2` reduces to classic discrete AdaBoost; matches
  `sklearn.ensemble.AdaBoostClassifier` (SAMME) closely on the
  deterministic stump path. No randomness, so no `random_state`; serial.
  `AdaBoostRegressor` and `SAMME.R` are out of scope. Derivation:
  `docs/derivations/adaboost.md`; example: `examples/adaboost.py`.
- `scratchgrad.tree.DecisionTreeClassifier.fit` gains an optional
  `sample_weight` argument — non-negative per-row weights that scale each
  row's contribution to the class counts and the impurity decrease (the
  `min_samples_*` gates still count rows). `None` is byte-identical to the
  previous fit; integer weights match a fit on the row-duplicated dataset;
  zero-weight rows are dropped, as scikit-learn does. This is the
  reweighting hook the M2 boosting ensembles (AdaBoost next) build on.
  Derivation: `docs/derivations/decision_tree.md` §3b.
- `scratchgrad.utils.check_sample_weight` — shared validator that coerces a
  `sample_weight` argument to a length-`n` float64 array (or synthesises
  all-ones), rejecting wrong shapes, negatives, and all-zero input.
- `scratchgrad.ensemble.RandomForestClassifier` — a random forest: an
  ensemble of `DecisionTreeClassifier`s, each grown on a bootstrap
  resample of the rows and drawing a random `max_features` subset
  (default `"sqrt"`) at every split. Prediction averages the trees'
  class-probability vectors (soft voting). `bootstrap` and `oob_score`
  are exposed; with `oob_score=True` the out-of-bag rows give
  `oob_score_` / `oob_decision_function_` for free. `random_state` seeds
  an independent `Generator` per tree, so the forest is reproducible.
  Trees are built serially (no `n_jobs`). Multiclass; held-out accuracy
  and OOB score track `sklearn.ensemble.RandomForestClassifier` to a
  tolerance (the RNG streams differ, so predictions are not exact). First
  `scratchgrad.ensemble` algorithm; first M2 algorithm. Derivation:
  `docs/derivations/random_forest.md`; example: `examples/random_forest.py`.
- `scratchgrad.tree.DecisionTreeClassifier` gains `max_features` (per-node
  feature subsampling — `"sqrt"`, `"log2"`, an int, a float fraction, or
  `None`) and `random_state` (seeds that draw), plus the resolved
  `max_features_` attribute. With the `None` defaults a fitted tree is
  byte-identical to the `0.1.0` estimator; the knobs exist so
  `RandomForest` (M2) can decorrelate its trees. Derivation:
  `docs/derivations/decision_tree.md` §3a.

## [0.1.0] — 2026-09-07

### Added

- `scratchgrad.linear.LinearRegression` — ordinary least squares with two
  solvers (`"normal"`: normal equation via SVD least-squares; `"gd"`: batch
  gradient descent). First M1 algorithm. Derivation:
  `docs/derivations/linear_regression.md`; example: `examples/linear_regression.py`.
- `scratchgrad.linear.Ridge` — L2-penalised least squares with two solvers
  (`"normal"`: regularised normal equation via `np.linalg.solve`; `"gd"`:
  batch gradient descent). `alpha` matches `sklearn.linear_model.Ridge`;
  `alpha=0` reduces to `LinearRegression`. Derivation:
  `docs/derivations/ridge.md`; example: `examples/ridge.py`.
- `scratchgrad.linear.Lasso` — L1-penalised least squares via cyclic
  coordinate descent with soft-thresholding (drives coefficients to exactly
  zero — variable selection). `alpha` matches `sklearn.linear_model.Lasso`
  (objective scaled by `1 / (2 * n_samples)`); `alpha=0` reduces to
  `LinearRegression`. Derivation: `docs/derivations/lasso.md`; example:
  `examples/lasso.py`.
- `scratchgrad.linear.LogisticRegression` — binary logistic regression by
  Bernoulli maximum likelihood (mean cross-entropy loss) with an optional L2
  penalty. Two iterative solvers: `"newton"` (Newton–Raphson / IRLS, the
  default — quadratic convergence, no learning rate) and `"gd"` (batch
  gradient descent). `C` matches `sklearn.linear_model.LogisticRegression`;
  `penalty=None` fits the unregularised MLE. `predict_proba`,
  `decision_function`, `score` = accuracy. Derivation:
  `docs/derivations/logistic_regression.md`; example:
  `examples/logistic_regression.py`.
- `scratchgrad.neighbors.KNeighborsClassifier` — lazy k-nearest-neighbours
  classification. `fit` memorises the data; `predict` builds the full
  distance matrix by brute force (no KD-/ball-tree), takes the `k` nearest,
  and votes. `metric` in `{"euclidean", "manhattan"}`, `weights` in
  `{"uniform", "distance"}` (1/d, with an exact-hit rule). `predict_proba`,
  `kneighbors`, `score` = accuracy; multiclass. First `scratchgrad.neighbors`
  algorithm. Derivation: `docs/derivations/knn.md`; example: `examples/knn.py`.
- `scratchgrad.naive_bayes.GaussianNB` — Gaussian naive Bayes, a generative
  classifier fitted by a single closed-form maximum-likelihood pass (per-class,
  per-feature mean and biased variance, plus class priors). Prediction is done
  in log-space: `predict` takes the argmax of the joint log-likelihood,
  `predict_proba` / `predict_log_proba` normalise it with `logsumexp`.
  `var_smoothing` adds `var_smoothing * max_j Var[X[:, j]]` to every variance;
  `priors` overrides the MLE class frequencies. Multiclass; matches
  `sklearn.naive_bayes.GaussianNB`. First `scratchgrad.naive_bayes` algorithm.
  Derivation: `docs/derivations/gaussian_nb.md`; example:
  `examples/gaussian_nb.py`.
- `scratchgrad.tree.DecisionTreeClassifier` — a CART decision tree grown by
  greedy recursive binary partitioning. Each node takes the `(feature,
  threshold)` split that maximises the impurity decrease, with impurity
  measured by `criterion="gini"` (`1 - sum p_k^2`) or `"entropy"` (Shannon
  entropy in bits). Thresholds are midpoints between adjacent distinct
  feature values; ties are broken deterministically (lowest feature index,
  then lowest threshold). Recursion stops at `max_depth`,
  `min_samples_split`, a pure node, or when no split clears
  `min_impurity_decrease` (scaled as `n_t / n * delta_H`, matching
  scikit-learn). `predict_proba` returns the leaf's class-frequency vector;
  `feature_importances_` is the normalised total impurity decrease per
  feature. Multiclass; matches `sklearn.tree.DecisionTreeClassifier`. No
  cost-complexity pruning or feature subsampling (`DecisionTreeRegressor`
  and `max_features` are later additions). First `scratchgrad.tree`
  algorithm; finishes M1. Derivation: `docs/derivations/decision_tree.md`;
  example: `examples/decision_tree.py`.
- `tests/reference/` and `.github/workflows/reference.yml` (manual trigger):
  opt-in scikit-learn parity tests, run with `pytest -m reference`.
- Repository scaffold: `pyproject.toml`, `.gitattributes`, `.gitignore`, `LICENSE` (MIT).
- `scratchgrad.base.Estimator`, `scratchgrad.typing`, `scratchgrad.exceptions`.
- `scratchgrad.utils`: input validation (`check_X_y`, `check_array`, `check_is_fitted`,
  `check_random_state`) and numerically stable elementary functions (`sigmoid`, `softmax`,
  `logsumexp`).
- `scratchgrad.metrics`: regression (`mean_squared_error`, `root_mean_squared_error`,
  `mean_absolute_error`, `r2_score`), classification (`accuracy_score`, `confusion_matrix`,
  `precision_score`, `recall_score`, `f1_score`), and pairwise distance/similarity
  (`euclidean_distance`, `manhattan_distance`, `cosine_similarity`).
- `scratchgrad.preprocessing`: `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`,
  `train_test_split`.
- `scratchgrad.datasets`: synthetic generators `make_regression`, `make_blobs`, `make_moons`.
- `tests/helpers/gradcheck.py`: finite-difference gradient checking, used by every
  backward pass from M3 onward.
- CI (`.github/workflows/ci.yml`), pre-commit config, PR template.
- Project documentation: `README.md`, `plan.md`, `ROADMAP.md`, `CONTRIBUTING.md`,
  `docs/conventions.md`.

<!-- Repo: https://github.com/unpopularchief/ML-From-Scratch
     [Unreleased] compare link added once the first commit/tag exists. -->
