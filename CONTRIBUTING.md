# Contributing

This project is built one phase at a time — see `plan.md` §0. This
document is the practical, day-to-day version of that plan.

## The workflow, per algorithm

1. **Derivation first, always.** Before writing any implementation code,
   the mathematical derivation is written up: notation (matching
   `docs/conventions.md`), the objective/loss function, the derivation
   itself (closed-form or gradient/update rule), pseudocode, and what a
   test will check. This gets confirmed *before* implementation starts,
   and becomes `docs/derivations/<algorithm>.md`.
2. Implement, following the conventions below.
3. Tests: analytic checks always; a gradient check
   (`tests/helpers/gradcheck.py`) for anything with a `backward()`;
   behavioral checks against synthetic data with known structure.
4. An example script in `examples/`, runnable end-to-end.
5. Update `README.md`'s algorithm table and `ROADMAP.md` in the *same*
   PR — not a follow-up cleanup pass.
6. One PR per algorithm (see `.github/PULL_REQUEST_TEMPLATE.md`).

## The NumPy line

NumPy is allowed for **array mechanics and linear-algebra primitives**
(`@`, `np.linalg.solve`, `np.linalg.svd`). It is **never** allowed to
implement the ML algorithm itself. Where a decomposition *is* the lesson
(PCA, SVD), ship a from-scratch version (power iteration, QR) *alongside*
the `np.linalg` one, with a test asserting they agree.

## What not to import

`scipy`, `pandas`, `numba`/`cython`, any GPU backend, `tensorflow`/`jax`.
See `plan.md` §6 for why each one would quietly remove a lesson (e.g.
`scipy.special.expit` is exactly what `scratchgrad.utils.math.sigmoid`
exists to avoid needing). Runtime dependencies are NumPy, full stop.
`scikit-learn`/`pytorch` are dev-only, for parity tests and benchmarks —
see `pyproject.toml`'s `[reference]` extra.

## Conventions

Full detail in `docs/conventions.md`. In short: `float64` everywhere,
scikit-learn-shaped `fit`/`predict`/`transform` API, trailing-underscore
learned attributes, explicit `random_state` (never the global NumPy RNG),
NumPy-style docstrings that state the governing equation, one
algorithm/layer per file.

## Mistakes this project has already decided to avoid

See `plan.md` §8 for the full, reasoned list. The two easiest to
backslide on:

- **Don't scaffold ahead.** A milestone's folders (`nn/`, `autograd/`,
  `attention/`, `transformer/`, `optim/`, and unstarted algorithm
  families) don't exist until that milestone starts.
- **Don't abstract before three examples exist.** No `BaseOptimizer`,
  `BaseKernel`, etc. until the shared shape is observed across at least
  three concrete implementations, not guessed in advance.
