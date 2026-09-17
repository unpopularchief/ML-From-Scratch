"""Tests for scratchgrad.nn.Module.

Tiers (plan.md section 3, docs/derivations/nn.md section 6 and
docs/derivations/dropout_batchnorm.md section 4): the default empty-list
parameters()/grads() contract, that forward/backward raise
NotImplementedError on the bare base class, and the training/eval flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Module


class TestBaseContract:
    def test_parameters_and_grads_default_to_empty_lists(self) -> None:
        module = Module()
        assert module.parameters() == []
        assert module.grads() == []

    def test_forward_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Module().forward(np.array([1.0]))

    def test_backward_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Module().backward(np.array([1.0]))


class TestTrainingFlag:
    def test_defaults_to_training_mode(self) -> None:
        assert Module().training is True

    def test_eval_sets_training_false_and_returns_self(self) -> None:
        module = Module()
        result = module.eval()
        assert module.training is False
        assert result is module

    def test_train_sets_training_true_and_returns_self(self) -> None:
        module = Module().eval()
        result = module.train()
        assert module.training is True
        assert result is module

    def test_flag_is_independent_per_instance(self) -> None:
        a, b = Module(), Module()
        a.eval()
        assert a.training is False
        assert b.training is True
