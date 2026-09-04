# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `scratchgrad.linear.LinearRegression` — ordinary least squares with two
  solvers (`"normal"`: normal equation via SVD least-squares; `"gd"`: batch
  gradient descent). First M1 algorithm. Derivation:
  `docs/derivations/linear_regression.md`; example: `examples/linear_regression.py`.
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

