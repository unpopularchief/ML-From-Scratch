"""Tests for scratchgrad.nn.Module.

Tiers (plan.md section 3, docs/derivations/nn.md section 6): the default
empty-list parameters()/grads() contract, and that forward/backward raise
NotImplementedError on the bare base class.
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
