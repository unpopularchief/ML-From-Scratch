# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
