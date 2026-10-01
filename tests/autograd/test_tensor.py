"""Tensor graph mechanics: accumulation, topological order, backward() contract."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.autograd import Tensor


class TestBasics:
    def test_data_is_float64(self):
        assert Tensor([1, 2, 3]).data.dtype == np.float64

    def test_grad_starts_none_and_zero_grad_resets(self):
        x = Tensor(2.0, requires_grad=True)
        assert x.grad is None
        (x * x).backward()
        assert x.grad is not None
        x.zero_grad()
        assert x.grad is None

    def test_detach_cuts_graph(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        d = (x * 2.0).detach()
        assert not d.requires_grad and d._parents == ()
        d_copy_check = x.detach()
        d_copy_check.data[0] = 99.0
        assert x.data[0] == 1.0

    def test_repr_and_shape(self):
        t = Tensor(np.zeros((2, 3)), requires_grad=True)
        assert t.shape == (2, 3)
        assert "requires_grad=True" in repr(t)

    def test_no_graph_recorded_when_nothing_requires_grad(self):
        out = Tensor([1.0]) * Tensor([2.0])
        assert not out.requires_grad and out._parents == ()


class TestBackwardContract:
    def test_scalar_seed_defaults_to_one(self):
        x = Tensor([3.0], requires_grad=True)
        (x * 2.0).backward()
        np.testing.assert_array_equal(x.grad, [2.0])

    def test_non_scalar_needs_explicit_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(ValueError, match="explicitly"):
            (x * 2.0).backward()

    def test_explicit_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        (x * 2.0).backward(np.array([1.0, 10.0]))
        np.testing.assert_array_equal(x.grad, [2.0, 20.0])

    def test_wrong_grad_shape_raises(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(ValueError, match="does not match"):
            (x * 2.0).backward(np.ones(3))

    def test_backward_on_constant_raises(self):
        with pytest.raises(ValueError, match="does not require grad"):
            Tensor(1.0).backward()

    def test_constants_get_no_grad(self):
        x = Tensor([1.0], requires_grad=True)
        c = Tensor([5.0])
        (x * c).sum().backward()
        assert c.grad is None


class TestFanOutAndAccumulation:
    def test_same_tensor_used_twice(self):
        x = Tensor([3.0], requires_grad=True)
        (x * x + x).sum().backward()  # d/dx (x^2 + x) = 2x + 1
        np.testing.assert_array_equal(x.grad, [7.0])

    def test_diamond_graph(self):
        # y = a*b with a = x+1, b = x+2  ->  dy/dx = a + b = 2x + 3
        x = Tensor([2.0], requires_grad=True)
        ((x + 1.0) * (x + 2.0)).sum().backward()
        np.testing.assert_array_equal(x.grad, [7.0])

    def test_leaf_grads_accumulate_across_backward_calls(self):
        x = Tensor([2.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        y.backward()
        np.testing.assert_array_equal(x.grad, [8.0])  # 2 * (2x)

    def test_second_backward_does_not_double_count_intermediates(self):
        # If intermediate grads were not reset, the second call would
        # propagate the stale first-call gradient too (giving 3x, not 2x).
        x = Tensor([1.0], requires_grad=True)
        y = ((x * 3.0) * 1.0).sum()
        y.backward()
        y.backward()
        np.testing.assert_array_equal(x.grad, [6.0])

    def test_zero_grad_then_backward_is_fresh(self):
        x = Tensor([2.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        x.zero_grad()
        y.backward()
        np.testing.assert_array_equal(x.grad, [4.0])


class TestTopologicalOrder:
    def test_parents_precede_children(self):
        x = Tensor([1.0], requires_grad=True)
        a = x * 2.0
        b = a + x
        c = (a * b).sum()
        order = c._topological_order()
        position = {id(n): i for i, n in enumerate(order)}
        for node in order:
            for p in node._parents:
                if p.requires_grad:
                    assert position[id(p)] < position[id(node)]

    def test_each_node_once(self):
        x = Tensor([1.0], requires_grad=True)
        a = x * 2.0
        c = (a + a + a).sum()
        order = c._topological_order()
        assert len(order) == len({id(n) for n in order})

    def test_deep_chain_does_not_hit_recursion_limit(self):
        x = Tensor([1.0], requires_grad=True)
        y = x
        for _ in range(5000):
            y = y + 1.0
        y.sum().backward()
        np.testing.assert_array_equal(x.grad, [1.0])


class TestComposite:
    def test_small_mlp_matches_hand_derived_gradients(self):
        # The nn/ lesson in miniature: y = tanh(X W + b), L = mean(y^2).
        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 3))
        W = Tensor(rng.standard_normal((3, 2)), requires_grad=True)
        b = Tensor(rng.standard_normal(2), requires_grad=True)

        y = (Tensor(X) @ W + b).tanh()
        (y**2).mean().backward()

        z = X @ W.data + b.data
        t = np.tanh(z)
        dz = (2.0 * t / t.size) * (1.0 - t**2)
        np.testing.assert_allclose(W.grad, X.T @ dz, atol=1e-12)
        np.testing.assert_allclose(b.grad, dz.sum(axis=0), atol=1e-12)
