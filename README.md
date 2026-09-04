# ML From Scratch

Machine learning and deep learning algorithms implemented from first
principles in Python and NumPy — no scikit-learn, no PyTorch, no
`scipy.special`. The goal is to see and understand the mathematics, not
to compete with production libraries on speed or features.

If you're looking for a fast, production-ready ML library, use
scikit-learn or PyTorch. If you want to see exactly how gradient descent,
backpropagation, and attention actually work, line by line, that's what
this is for.

## Status

🚧 **M1 in progress.** The M0 foundation is in place (validation utilities,
metrics, preprocessing transforms, dataset generators, and the
gradient-checking test tool). M1 adds the classical supervised algorithms
one at a time — `LinearRegression` and `Ridge` are implemented so far; see
the table below. See [`ROADMAP.md`](ROADMAP.md) for what's planned and in what
order, and
[`plan.md`](plan.md) for the full project plan (architecture, testing
strategy, conventions, and the mistakes it's deliberately avoiding).

## Philosophy

- **Readability of the derivation beats performance**, always. A slower
  implementation that maps clearly onto the math it's implementing is a
  win here; a fast one-liner nobody can trace back to an equation is not.
- **NumPy only.** `scipy`, `pandas`, and GPU backends are deliberately
  out — see `plan.md` §6 for why each one would quietly remove a lesson.
- **Every backward pass is gradient-checked** against numerical
  (finite-difference) gradients, not just "it looks right."
- **We are 10–100× slower than scikit-learn**, and that's expected: their
  inner loops are Cython/BLAS, ours are readable NumPy. Publishing that
  gap honestly (`benchmarks/`) is itself part of the point.

## Installation

Requires Python ≥3.10. Managed with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync --extra dev
```

## Development

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format check
uv run pytest -q             # tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow, including
the derivation-before-code process every algorithm goes through.
Contributions are welcome — pick anything unchecked in
[`ROADMAP.md`](ROADMAP.md), or open an issue or PR with an idea. New
collaborators are welcome too; if you'd like to help maintain the project,
just reach out.

## Algorithms

One row per algorithm as each ships. See [`ROADMAP.md`](ROADMAP.md) for
what's planned next.

| Algorithm        | Module               | Derivation                                                    | Notes                                                    |
| ---------------- | -------------------- | ------------------------------------------------------------- | ------------------------------------------------------- |
| LinearRegression | `scratchgrad.linear` | [linear_regression.md](docs/derivations/linear_regression.md) | OLS; normal equation and batch gradient descent         |
| Ridge            | `scratchgrad.linear` | [ridge.md](docs/derivations/ridge.md)                         | L2-penalised least squares; regularised normal equation and gradient descent |

## License

MIT — see [`LICENSE`](LICENSE). Free to use, copy, modify, and redistribute,
for any purpose, including commercially. Use it in your own projects, fork
it, teach from it — no permission needed.

Started and maintained by **Niyaf Mukhthaar**. Open to anyone who wants
to join in and build it out.
