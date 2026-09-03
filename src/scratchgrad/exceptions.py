"""Project-wide exceptions and warnings."""

from __future__ import annotations


class NotFittedError(ValueError, AttributeError):
    """Raised when ``predict``/``transform`` is called before ``fit``.

    Inherits from both ``ValueError`` and ``AttributeError`` so it can be
    caught either way, matching the convention scikit-learn uses for the
    same situation.
    """


class ConvergenceWarning(UserWarning):
    """Raised when an iterative estimator stops before converging.

    Signals that ``n_iter_`` reached ``max_iter`` without the stopping
    criterion (e.g. gradient norm, parameter change) being met.
    """
