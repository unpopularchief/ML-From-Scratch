"""Tests for scratchgrad.base.Estimator."""

from __future__ import annotations

from scratchgrad.base import Estimator


class _NoInit(Estimator):
    """A subclass that defines no __init__ of its own."""


class _WithParams(Estimator):
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept


def test_get_params_empty_for_subclass_without_init() -> None:
    # Regression test: object.__init__'s signature is (*args, **kwargs),
    # which must not be reported as if they were real hyperparameters.
    assert _NoInit().get_params() == {}


def test_get_params_reads_constructor_arguments() -> None:
    estimator = _WithParams(alpha=0.5, fit_intercept=False)
    assert estimator.get_params() == {"alpha": 0.5, "fit_intercept": False}


def test_get_params_reflects_current_attribute_value() -> None:
    # get_params reads live attributes, not the values passed at construction,
    # so mutating a stored hyperparameter after the fact is reflected too.
    estimator = _WithParams(alpha=0.5)
    estimator.alpha = 2.0
    assert estimator.get_params()["alpha"] == 2.0


def test_repr_contains_class_name_and_params() -> None:
    text = repr(_WithParams(alpha=0.5, fit_intercept=False))
    assert text == "_WithParams(alpha=0.5, fit_intercept=False)"


def test_repr_empty_params_for_subclass_without_init() -> None:
    assert repr(_NoInit()) == "_NoInit()"
