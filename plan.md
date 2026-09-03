# ML From Scratch — Project Plan

## Context

This repository is currently empty. This is a greenfield open-source project.

**Goal:** implement machine learning and deep learning algorithms from first principles in Python + NumPy, so the mathematics is visible in the code rather than hidden behind scikit-learn or PyTorch. The audience is people learning the math (including the author), so **readability of the derivation beats performance** at every decision point.

**Non-goal:** competing with scikit-learn or PyTorch on speed, features, or generality. Anything that trades clarity for throughput is out of scope.

**Verified environment:** Python 3.14.3, NumPy 2.5.1, git 2.54, `uv`, and `gh` are all installed and on PATH. PyTorch 2.14 and NumPy both ship 3.14 wheels, so the whole toolchain runs on one interpreter — no split environment needed.

### Decisions already made

| Decision | Choice |
| --- | --- |
| Package / import name | `scratchgrad` — verified free on PyPI (`mlscratch`, `scratchml`, `mlfs` are all taken) |
| Backprop approach | **Staged**: manual per-layer `backward()` for MLP/CNN/RNN, then a `Tensor` autograd engine as its own milestone, then attention/transformers on top of it |
| Testing | **Hybrid**: analytic + finite-difference gradient checks by default; opt-in scikit-learn/PyTorch parity tests behind a marker |
| Teaching material | Runnable `.py` scripts in `examples/` + math derivations in docstrings and `docs/derivations/`. **No notebooks in git.** |
| dtype policy | **`float64` everywhere**, no exceptions, until a milestone produces a concrete, demonstrated reason to do otherwise |

---

## 0. Execution process — phased, not all at once

This project is **not** built by scaffolding the whole tree and then filling it in. It proceeds one phase at a time, and each phase is small enough to fully verify before the next one starts.

1. **Milestone by milestone.** M0 → M1 → … → M7, in order (§7). A later milestone's folders (`nn/`, `autograd/`, `attention/`, `transformer/`, `optim/`, and every algorithm family under it) are **not created** until that milestone actually begins. No empty directories or stub files placed ahead of need — if a folder exists, it's because something in it is being worked on now.
2. **Algorithm by algorithm within a milestone.** Each algorithm (or layer, in the nn/autograd milestones) is its own unit of work: derivation → implementation → tests → example → docs, reviewed and confirmed before moving to the next algorithm in the same milestone.
3. **Derivation before code — every algorithm, no exceptions.** Before writing a line of implementation for any algorithm (starting with M1's LinearRegression, all the way through the transformer), the mathematical derivation is walked through in chat first:
   - the problem setup and notation used (tied to `docs/conventions.md`)
   - the objective/loss function being optimized
   - the derivation itself — closed-form solution, or the gradient/update rule, worked from the objective
   - the resulting algorithm stated as pseudocode
   - what a test will check to confirm the implementation matches the derivation (e.g., "gradient checked against central finite differences", "matches the normal equation on a hand-computed 3-point dataset")

   Implementation starts only after that derivation is confirmed. The finalized version is what becomes the algorithm's `docs/derivations/*.md` file.
4. **Verify before advancing.** A phase is done when its tests are green and (if it has one) its example script runs end-to-end. That gets confirmed before starting the next phase — milestones are not parallelized and not preemptively built ahead.
5. **This plan file is the only thing being written right now.** No scaffolding, no dependency installs, no `git init`, until the plan itself is approved.

---

## 1. Repository architecture

`src/` layout (keeps tests honest — they import the installed package, not the working directory).

**This is the destination, not the M0 checklist.** Per §0, only the folders needed by the *current* milestone exist at any time. The tree below shows where things end up by M7 — `linear/`, `tree/`, `svm/`, `nn/`, `autograd/`, `attention/`, `transformer/`, `optim/`, and `tests/reference/` are each created when their own milestone starts, not now.

```
ML From Scratch/
├── src/scratchgrad/
│   ├── __init__.py            # curated public API, __version__
│   ├── base.py                # Estimator ABC: fit/predict/score, get_params, __repr__
│   ├── typing.py              # Float64Array / Float32Array aliases
│   ├── exceptions.py          # NotFittedError, ConvergenceWarning
│   ├── utils/
│   │   ├── validation.py      # check_X_y, check_is_fitted, check_random_state
│   │   └── math.py            # sigmoid, softmax, logsumexp (stable, hand-written)
│   ├── datasets/              # make_blobs/moons/regression + MNIST & tiny-shakespeare loaders
│   ├── metrics/               # classification.py, regression.py, pairwise.py
│   ├── preprocessing/         # StandardScaler, MinMaxScaler, OneHotEncoder, train_test_split
│   ├── optim/                 # sgd, momentum, nesterov, rmsprop, adam (shared: classical + nn)
│   │
│   ├── linear/                # linear_regression, ridge, lasso, logistic_regression
│   ├── neighbors/             # knn
│   ├── naive_bayes/           # gaussian, multinomial
│   ├── tree/                  # decision_tree.py (CART) — one file; split out a criterion/
│   │                          #   or splitter module only if it later grows unwieldy
│   ├── ensemble/              # random_forest, adaboost, gradient_boosting
│   ├── svm/                   # linear_svm (hinge + subgradient), kernel_svm (SMO, optional)
│   ├── cluster/               # kmeans, dbscan, gaussian_mixture
│   ├── decomposition/         # pca, svd
│   │
│   ├── nn/                    # PHASE 2–3: explicit, hand-derived backward passes
│   │   ├── module.py          # Module base: forward(x) / backward(grad) / parameters()
│   │   ├── layers/            # linear, conv2d, pooling, flatten, dropout, batchnorm,
│   │   │                      #   rnn, lstm
│   │   ├── activations.py     # relu, sigmoid, tanh, softmax — each with its own backward
│   │   ├── losses.py          # mse, cross_entropy, bce
│   │   ├── init.py            # xavier, he, zeros
│   │   └── trainer.py         # minimal fit loop: batching, epochs, history
│   │
│   ├── autograd/              # PHASE 4: the engine
│   │   ├── tensor.py          # Tensor(data, requires_grad), topological backward()
│   │   ├── ops.py             # primitive ops + their vector-Jacobian products
│   │   ├── functional.py      # softmax, cross_entropy, gelu, layer_norm
│   │   ├── layers.py          # Linear, LayerNorm, Embedding, Dropout — on Tensor
│   │   └── optim.py           # Adam / AdamW over Tensor parameters
│   │
│   ├── attention/             # PHASE 5: built on autograd
│   │   ├── scaled_dot_product.py
│   │   ├── masking.py
│   │   └── multi_head.py
│   └── transformer/           # PHASE 6: capstone
│       ├── positional.py      # sinusoidal + learned
│       ├── block.py           # pre-LN residual block
│       ├── encoder.py / decoder.py
│       ├── tokenizer.py       # char-level, then a minimal BPE
│       └── gpt.py             # tiny char-level GPT
│
├── tests/                     # mirrors src/ package-for-package
│   ├── conftest.py            # shared `rng` fixture, tolerance constants
│   ├── helpers/gradcheck.py   # central finite differences — the project's key test tool
│   └── reference/             # @pytest.mark.reference — sklearn/torch parity, opt-in
│
├── examples/                  # runnable .py scripts, each < ~80 lines, seeded
├── benchmarks/
│   └── run.py                 # single script, prints a comparison table to the console
│                               # (results/ + docs/benchmarks.md added later, once there's
│                               #  an actual algorithm and a reader for them — see §4)
├── docs/
│   ├── conventions.md         # notation table + API contract
│   └── derivations/           # one .md per algorithm: the math, properly written out
├── .github/
│   ├── workflows/ci.yml        # reference.yml added at M1, once there's something to compare
│   └── PULL_REQUEST_TEMPLATE.md
├── pyproject.toml
├── README.md · ROADMAP.md · CHANGELOG.md · CONTRIBUTING.md · LICENSE (MIT)
├── .gitignore · .gitattributes · .pre-commit-config.yaml
```

**Why `nn/` and `autograd/` coexist:** they are two tellings of the same story. `nn/` shows the chain rule written by hand; `autograd/` shows it automated. A test asserts they produce identical gradients for the same MLP — that equality *is* the lesson, so the duplication is deliberate, not debt.

### Public API contract

Deliberately scikit-learn-shaped. It's a real convention worth learning, and it makes reference benchmarking almost free.

- `fit(X, y) -> self`, `predict(X)`, `predict_proba(X)`, `score(X, y)`, `transform` / `fit_transform`
- Hyperparameters are `__init__` arguments and are **never mutated** by `fit`
- Learned attributes carry a trailing underscore: `coef_`, `intercept_`, `n_iter_`, `labels_`
- Randomness always via an explicit `random_state` → `np.random.default_rng`; the global NumPy RNG is never touched
- `base.Estimator` stays under ~60 lines: `get_params`, `__repr__`, `check_is_fitted`. No cloning, no meta-estimator machinery, no validation framework.

---

## 2. Coding & documentation conventions

- **Docstrings:** NumPy style. Every estimator documents Parameters, Attributes, Notes (the math), References (book/paper + chapter), Examples.
- **The math goes in the code.** The module docstring carries the governing equation; inline comments name the term each line implements. The rule for reviewers: *if a line computes a term in an equation, the comment names that term.*
- **Consistent notation** across the whole codebase, recorded in `docs/conventions.md`: `X` is `(n_samples, n_features)`, `y` is `(n_samples,)`, `W` weights, `b` bias, `n` samples, `d` features, `lr` learning rate.
- **Vectorize, but never at the cost of the math.** Where a vectorized expression obscures the derivation, keep the naive loop in the docstring as a `# reference implementation` block and test the two against each other.
- **One algorithm per file.** No `models.py`.
- **Type hints on all public signatures**, using the aliases in `typing.py`.
- **dtype policy:** `float64` everywhere, no split by module. Enforced in `check_X_y`, stated in `conventions.md`. (A dual float32/float64 policy was considered and dropped — see §8, mistake 15.)
- **Tooling:** `ruff` for both lint and format (replaces black + isort + flake8). Line length 88. Rules: `E, F, I, UP, B, NPY, D`. No `mypy` for now — one linter is enough surface for a project this size; add static typing tooling later only if untyped bugs actually start showing up.

---

## 3. Testing strategy

`pytest`, `tests/` mirroring `src/`, four tiers:

1. **Analytic** — closed-form checks on tiny hand-computed data. OLS against the normal equation; entropy of a known split; PCA of a known covariance matrix. Plus shape/dtype contracts and edge cases (one sample, constant feature, single-class target, empty cluster).
2. **Gradient checks** — central finite differences vs. the analytic gradient, for **every** layer, activation, and loss. `tests/helpers/gradcheck.py` is written in M0 and is the highest-value test in the project: it is what makes hand-derived backprop trustworthy.
3. **Behavioral** — on seeded synthetic data with known structure, assert the model reaches a loss/accuracy threshold. Catches "gradients are right but the optimizer never converges".
4. **Reference parity** — `@pytest.mark.reference`, compares coefficients/predictions to scikit-learn and PyTorch within tolerance. **Deselected by default** via `addopts = "-m 'not reference'"`, run in its own CI workflow.

Determinism is non-negotiable: every test seeds explicitly, no test depends on global RNG state. Coverage target ~90% on `src/`, tracked but not enforced as a merge gate until after v0.1.0.

`hypothesis` property tests (scaler round-trips, softmax sums to 1, distance-metric symmetry) get added only where they earn their place — not upfront.

---

## 4. Benchmarking

Separate from tests, no CI gate, run manually. Kept intentionally minimal — this is a script, not a pipeline: **not built until M1 ships an algorithm to benchmark.**

`benchmarks/run.py`: a single script that, for a given algorithm, times ours against scikit-learn's on the same synthetic data and prints one table to the console (`time.perf_counter`, median of N runs, fixed seeds). It checks two things:

- **Correctness parity** — do our coefficients/predictions match scikit-learn's within tolerance, on real-shaped data?
- **Performance ratio** — wall-clock across growing `n_samples`/`n_features`, reported as a multiple of scikit-learn's time.

No committed JSON snapshots, no results-history, no auto-generated `docs/benchmarks.md` — that's a doc-generation pipeline with no reader yet. If tracking results over time turns out to matter later, add it then, not speculatively now. Numbers worth keeping get pasted into the README by hand.

**The README states the honest framing up front:** we are 10–100× slower than scikit-learn, that is the expected outcome, and here is why (their inner loops are Cython/BLAS, ours are readable NumPy). Publishing that gap *is* educational content. Chasing it would be a mistake.

---

## 5. Git / GitHub workflow

- `git init`, MIT license, `main` as the default branch.
- **`.gitattributes` with `* text=auto eol=lf` from the very first commit** — development is on Windows, and this prevents CRLF noise polluting every future diff and PR.
- Short-lived branches: `feat/logistic-regression`, `fix/…`, `docs/…`. Conventional Commits. Squash-merge to keep `main` linear and readable.
- **One PR per algorithm** — this is the natural unit from §0's phased process (derivation → code → tests → example, confirmed, then merged), not extra ceremony layered on top of it. Template checklist: derivation doc · unit tests · gradient check (if it has a backward pass) · example script · README algorithm-table row.
- **CI** (`ci.yml`): `ruff check` + `ruff format --check` + `pytest`. One Python version (3.12) on ubuntu-latest and windows-latest to start — Windows matters because development happens there (catches CRLF/path bugs early); a multi-version Python matrix is added later once there's enough code for cross-version bugs to be a real risk, not now.
- **`reference.yml`** is **not created at M0.** It's added at M1, when `[reference]` first has scikit-learn parity tests to actually run. Manual trigger only — no nightly schedule; there are no long-running regressions to catch yet, and a scheduled workflow for a project with one contributor is a notification nobody needs.
- `ROADMAP.md` is the tracker, with `good first issue` labels per unimplemented algorithm. No GitHub Projects board — unnecessary ceremony for a solo start.
- Tag `v0.x.0` at each milestone; `CHANGELOG.md` in Keep-a-Changelog format.
- `pre-commit`: ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-added-large-files.

---

## 6. Package structure & dependencies

Build backend **hatchling**, environment managed with **`uv`** (already installed). `requires-python = ">=3.10"`.

```toml
dependencies = ["numpy>=1.24"]           # runtime: NumPy and nothing else. Hard rule.

[project.optional-dependencies]
dev       = ["pytest", "pytest-cov", "ruff", "pre-commit"]
reference = ["scikit-learn"]             # parity tests only — torch added at M3, when
                                          # nn/autograd parity tests first need it
examples  = ["matplotlib"]               # plotting in examples/ only
```

### Use

NumPy (arrays + `np.linalg` as a *tool*), pytest, ruff, hatchling, uv, matplotlib (examples only), scikit-learn (comparison, from M1), PyTorch (comparison, from M3 only).

### Avoid — and why

- **scipy** — the single biggest temptation. `scipy.special.expit`, `scipy.optimize`, `scipy.spatial.distance` would each silently delete a lesson. Write them yourself in `utils/math.py`.
- **pandas** — adds a heavy dependency and a second array API for no pedagogical gain. NumPy arrays throughout.
- **Cython / Numba / C extensions** — speed at the direct cost of the one thing this project sells.
- **CuPy / any GPU backend** — a device-abstraction layer would dominate the codebase.
- **TensorFlow / JAX**, **Poetry** (uv + hatchling is lighter), **Sphinx** early (markdown in `docs/` first; MkDocs later only if the project outgrows it).

### The NumPy line — stated explicitly in CONTRIBUTING.md

NumPy is allowed for **array mechanics and linear-algebra primitives** (`@`, `np.linalg.solve`, `np.linalg.svd`). It is **never** allowed to implement the ML algorithm itself. Where a decomposition *is* the lesson — PCA, SVD — ship a from-scratch version (power iteration, QR) **alongside** the `np.linalg` one and add a test asserting they agree. That test teaches more than either implementation alone.

---

## 7. Progression & milestones

Milestones are sized in PRs, not calendar dates.

| # | Milestone | Contents | Tag |
| --- | --- | --- | --- |
| **M0** | **Foundation** | Repo scaffold, pyproject, CI, pre-commit, `base.py`, `utils/`, `metrics/`, `preprocessing/`, `datasets/` generators, **`gradcheck.py`**, docs skeleton, README. No algorithms yet. | `v0.0.1` |
| **M1** | **Classical supervised** | LinearRegression (normal equation *and* gradient descent), Ridge, Lasso (coordinate descent), LogisticRegression, KNN, GaussianNB, DecisionTree (CART: gini/entropy/MSE) | `v0.1.0` |
| **M2** | **Ensembles & unsupervised** | RandomForest, AdaBoost, GradientBoosting, KMeans, DBSCAN, GaussianMixture (EM), PCA, LinearSVM (kernel SVM/SMO optional) | `v0.2.0` |
| **M3** | **Optimizers + MLP** *(manual backward)* | `optim/` (SGD, momentum, Nesterov, RMSprop, Adam), `nn.Module`, Linear, activations, losses, init schemes, Dropout, BatchNorm, trainer. Full gradcheck coverage. MNIST example. | `v0.3.0` |
| **M4** | **CNN & RNN** *(still manual backward)* | Conv2d (im2col), MaxPool, Flatten, RNN cell, LSTM, BPTT. Examples: MNIST CNN, char-level RNN. | `v0.4.0` |
| **M5** | **Autograd engine** | `Tensor`, topological `backward()`, ops + VJPs, functional, Tensor-based layers and optimizers. **Proof test:** re-implement M3's MLP on autograd and assert its gradients match the hand-derived ones. | `v0.5.0` |
| **M6** | **Attention** | Scaled dot-product attention, causal + padding masks, multi-head attention, positional encodings. | `v0.6.0` |
| **M7** | **Transformer** *(capstone)* | Pre-LN block, encoder, decoder, char-level tokenizer then minimal BPE, tiny GPT trained on tiny-shakespeare, plus a full write-up. | `v1.0.0` |

### MVP — the first thing worth publishing

**M0 + LinearRegression + LogisticRegression + KNN + metrics + one example + green CI + a real README.**

The MVP is *depth of quality on three algorithms*, not breadth across twenty. Every one of the three ships with its derivation doc, full tests, and a runnable example. That sets the quality bar the remaining ~25 algorithms must clear — and it's far more persuasive to a GitHub visitor than a wide, shallow, half-tested collection.

---

## 8. Engineering mistakes to actively avoid

Recorded in `CONTRIBUTING.md` so they survive past the first month:

1. **Building the autograd engine early.** It would erase the hand-derived backprop lesson, which is the point of M3–M4. Already settled: it arrives at M5.
2. **Premature abstraction.** No `BaseOptimizer` hierarchy until three optimizers exist and the shared shape is *observed*, not guessed. Same for kernels, criteria, and layers.
3. **A registry / plugin system** for estimators. Twenty-five algorithms in one repo need imports, not a framework.
4. **YAML config files and an experiment runner.** `examples/` are scripts with hardcoded, visible hyperparameters. Configs hide exactly what a learner needs to see.
5. **Chasing scikit-learn's performance.** Publish the gap, explain it, move on.
6. **Over-vectorizing past readability.** A clever one-line einsum that nobody can map back to the equation is a regression here, not an optimization.
7. **Deep inheritance chains.** Composition plus one shallow ABC. `Estimator → LinearModel → RegularizedLinearModel → Ridge` is exactly the trap to avoid.
8. **Reaching for scipy** the moment a special function is needed. See §6.
9. **Silent dtype drift** — float32/float64 mixing produces gradient-check failures that cost hours to diagnose. This is why §0/§2 settled on `float64` everywhere with no per-module split: one dtype means this class of bug can't happen. Enforce it in `check_X_y`.
10. **Committing datasets.** MNIST and tiny-shakespeare download to a gitignored cache dir on first use.
11. **Skipping gradient checks** because a layer "looks right". Every backward pass gets one; the PR template enforces it.
12. **A monolithic `nn/layers.py`.** By M7 it would be thousands of lines. One layer per file from the start.
13. **Documentation drift** — the README algorithm table and `ROADMAP.md` are updated in the *same PR* as the implementation, never in a catch-up pass.
14. **Windows CRLF pollution.** `.gitattributes` in the very first commit.
15. **Scaffolding the whole package tree upfront.** Creating `linear/`, `tree/`, `nn/`, `autograd/`, `attention/`, `transformer/`, `optim/` as empty directories at M0 invites drift between "planned shape" and "actual shape" and gives false progress with no code behind it. Per §0, each is created only when its milestone starts.
16. **Deciding a dual dtype policy (or any other module-specific rule) before a concrete need shows up.** The original draft of this plan split `float64`/`float32` by module up front — a rule to remember and enforce before a single layer existed. Corrected to one dtype everywhere (§0, §2); revisit only if a real, demonstrated need appears.
17. **Building benchmark result-tracking and doc-generation pipelines before there's a single algorithm to benchmark.** `benchmarks/run.py` starts as one script that prints a table; snapshot storage and auto-generated docs are infrastructure for a problem (tracking benchmark history) that doesn't exist yet.

---

## Deliverables of the first implementation session (M0)

Files created, in order:

Per §0, M0 itself is also broken into small confirmed steps rather than dumped in one commit — roughly: (a) repo meta files + `git init`, (b) `pyproject.toml`, (c) package skeleton (`base`/`typing`/`exceptions`/`utils`), (d) `metrics`/`preprocessing`/`datasets`, (e) `gradcheck.py` + tests for all of the above, (f) CI/docs/README. Nothing in M0 implements an algorithm — no derivation discussion is needed for it. `optim/`, and every algorithm-family folder (`linear/`, `tree/`, `nn/`, `autograd/`, …), are **not** part of M0; they arrive with their own milestone.

Files created:

1. `.gitattributes`, `.gitignore`, `LICENSE` (MIT), then `git init` + first commit
2. `pyproject.toml` (hatchling, src layout, `scratchgrad`, `dev`/`examples` extras, ruff + pytest config — no `reference` extra yet, nothing to compare against until M1)
3. `src/scratchgrad/{__init__,base,typing,exceptions}.py`
4. `src/scratchgrad/utils/{validation,math}.py`
5. `src/scratchgrad/{metrics,preprocessing,datasets}/`
6. `tests/conftest.py`, `tests/helpers/gradcheck.py`, and tests for everything above
7. `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.github/PULL_REQUEST_TEMPLATE.md`
8. `README.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/conventions.md`

## Verification

M0 is done when all of the following pass locally on Windows:

```powershell
uv sync --extra dev
uv run ruff check . ; uv run ruff format --check .
uv run pytest -q
uv run python -c "import scratchgrad; print(scratchgrad.__version__)"
```

Then push to GitHub via `gh repo create` and confirm CI is green on both ubuntu-latest and windows-latest.

At M1, `[reference]` (scikit-learn) and `reference.yml` are added, and `uv run pytest -m reference -q` becomes part of verification from then on.

Each later milestone adds: every new backward pass has a passing `gradcheck` test, its example script runs end-to-end, and its derivation doc exists.
