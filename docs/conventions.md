# Conventions

The notation and API contract every module in this project follows. See
`plan.md` for the reasoning behind these choices.

## Notation

| Symbol | Meaning |
| --- | --- |
| `X` | Design matrix, shape `(n_samples, n_features)` |
| `y` | Target vector, shape `(n_samples,)` — labels or regression targets |
| `n`, `n_samples` | Number of samples |
| `d`, `n_features` | Number of features |
| `W` | Weight matrix/vector |
| `b` | Bias / intercept |
| `lr` | Learning rate |
| `y_hat`, `ŷ` | Predicted value |

Docstrings and derivation docs use this notation consistently so a reader
doesn't have to re-learn symbols moving between files.

## dtype policy

**`float64` everywhere, no exceptions.** Every array entering the library
is coerced to `float64` by `check_X_y`/`check_array`
(`scratchgrad.utils.validation`). No module uses `float32` — a dual dtype
policy was considered and dropped (see `plan.md` §0, §8 mistake 16); one
dtype eliminates a whole class of silent dtype-mismatch bugs.

## Public API contract

Every estimator (in the scikit-learn sense: something with `fit`) follows:

- `fit(X, y) -> self` (or `fit(X) -> self` for unsupervised estimators)
- `predict(X)`, `predict_proba(X)` where applicable
- `transform(X)` / `fit_transform(X)` for preprocessing steps
- `score(X, y)` where a natural default metric exists
- Hyperparameters are `__init__` arguments, stored under the same name as
  an instance attribute, and are **never mutated** by `fit`
- Learned attributes carry a trailing underscore: `coef_`, `intercept_`,
  `mean_`, `labels_`, `n_iter_`
- Randomness always flows through an explicit `random_state` argument →
  `scratchgrad.utils.validation.check_random_state`. The global NumPy RNG
  is never touched.

`scratchgrad.base.Estimator` provides `get_params()`/`__repr__()` from the
constructor signature — see its docstring for the one requirement this
places on subclasses (store each hyperparameter under its own name).

## Docstrings

NumPy style. Every public function/class documents Parameters, Returns
(or Attributes, for a class), and Raises where relevant. Where a function
implements a specific equation, the docstring states that equation and
inline comments name the term each line of code computes — the rule is:
*if a line computes a term in an equation, the comment names that term.*

## Vectorization

Vectorize, but never at the cost of obscuring the math. If a vectorized
expression would be unreadable next to its equation, keep a clear,
possibly-slower version and explain the trick in a comment (e.g. the
squared-norm expansion in `metrics.pairwise.euclidean_distance`) rather
than leaving a reader to reverse-engineer a one-liner.

## One thing per file

One algorithm/layer/transformer per file. No `models.py` grab-bags.

## Line length & tooling

88 columns, enforced by `ruff format`. Lint rules: `E, F, I, UP, B, NPY,
D` (see `pyproject.toml`).
