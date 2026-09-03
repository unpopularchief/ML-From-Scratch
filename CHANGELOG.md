# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

No algorithms yet — this is the M0 foundation milestone.

<!-- Repo: https://github.com/unpopularchief/ML-From-Scratch
     [Unreleased] compare link added once the first commit/tag exists. -->

