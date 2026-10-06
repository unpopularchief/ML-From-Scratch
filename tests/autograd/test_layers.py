"""Tensor layers: parity with the hand-derived nn layers, modes, validation."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.autograd import Tensor, ops
from scratchgrad.autograd.layers import (
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
from scratchgrad.nn import Dropout as NnDropout
from scratchgrad.nn import Linear as NnLinear

rng = np.random.default_rng(0)
X = rng.standard_normal((6, 3))
G = rng.standard_normal((6, 2))


class TestLinear:
    @pytest.mark.parametrize("init", ["he", "xavier", "zeros"])
    def test_matches_hand_derived_nn_linear(self, init: str) -> None:
        ref = NnLinear(3, 2, weight_init=init, random_state=7)
        layer = Linear(3, 2, weight_init=init, random_state=7)
        np.testing.assert_array_equal(layer.W.data, ref.W)

        x = Tensor(X, requires_grad=True)
        y = layer(x)
        np.testing.assert_allclose(y.data, ref.forward(X), atol=1e-12)

        y.backward(G)
        dX = ref.backward(G)
        np.testing.assert_allclose(x.grad, dX, atol=1e-10)
        np.testing.assert_allclose(layer.W.grad, ref.grads()[0], atol=1e-10)
        np.testing.assert_allclose(layer.b.grad, ref.grads()[1], atol=1e-10)

    def test_parameters_and_zero_grad(self) -> None:
        layer = Linear(3, 2, random_state=0)
        assert layer.parameters() == [layer.W, layer.b]
        layer(Tensor(X)).sum().backward()
        assert layer.W.grad is not None
        layer.zero_grad()
        assert layer.W.grad is None and layer.b.grad is None

    @pytest.mark.parametrize(
        "kwargs",
        [{"in_features": 0, "out_features": 1}, {"in_features": 1, "out_features": 0}],
    )
    def test_rejects_nonpositive_sizes(self, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Linear(**kwargs)

    def test_rejects_unknown_init(self) -> None:
        with pytest.raises(ValueError, match="weight_init"):
            Linear(2, 2, weight_init="bogus")


class TestDropout:
    def test_matches_nn_dropout_forward_and_backward(self) -> None:
        ref = NnDropout(p=0.4, random_state=3)
        layer = Dropout(p=0.4, random_state=3)
        x = Tensor(X, requires_grad=True)
        y = layer(x)
        np.testing.assert_allclose(y.data, ref.forward(X))
        y.backward(G[:, :1] * np.ones_like(X))
        np.testing.assert_allclose(x.grad, ref.backward(G[:, :1] * np.ones_like(X)))

    def test_eval_is_identity(self) -> None:
        x = Tensor(X)
        assert Dropout(0.5, random_state=0).eval()(x) is x

    def test_has_no_parameters(self) -> None:
        assert Dropout(0.5).parameters() == []

    def test_rejects_bad_p(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            Dropout(p=1.0)


class TestModuleAndSequential:
    def test_base_forward_is_abstract(self) -> None:
        with pytest.raises(NotImplementedError):
            Module()(Tensor(X))

    def test_sequential_composes_modules_and_callables(self) -> None:
        a, b = Linear(3, 4, random_state=0), Linear(4, 2, random_state=1)
        net = Sequential(a, ops.relu, b)
        expected = b(ops.relu(a(Tensor(X))))
        np.testing.assert_array_equal(net(Tensor(X)).data, expected.data)
        assert net.parameters() == [a.W, a.b, b.W, b.b]

    def test_train_eval_propagate_to_children(self) -> None:
        drop = Dropout(0.5, random_state=0)
        net = Sequential(Linear(3, 3, random_state=0), ops.relu, drop)
        assert net.eval() is net and not drop.training and not net.training
        assert net.train() is net and drop.training and net.training

    def test_zero_grad_clears_every_parameter(self) -> None:
        net = Sequential(Linear(3, 4, random_state=0), ops.tanh, Linear(4, 1))
        net(Tensor(X)).sum().backward()
        net.zero_grad()
        assert all(p.grad is None for p in net.parameters())


class TestLayerNorm:
    def test_initial_parameters_and_shape(self) -> None:
        layer = LayerNorm(4)
        np.testing.assert_array_equal(layer.gamma.data, np.ones(4))
        np.testing.assert_array_equal(layer.beta.data, np.zeros(4))
        assert layer(Tensor(rng.standard_normal((2, 3, 4)))).shape == (2, 3, 4)
        assert len(layer.parameters()) == 2

    def test_parameters_receive_gradients(self) -> None:
        layer = LayerNorm(3)
        layer(Tensor(X)).sum().backward()
        assert all(p.grad is not None for p in layer.parameters())

    def test_nonpositive_features(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LayerNorm(0)


class TestEmbeddingValidation:
    @pytest.mark.parametrize("sizes", [(0, 4), (4, 0)])
    def test_nonpositive_sizes(self, sizes: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="positive"):
            Embedding(*sizes)
