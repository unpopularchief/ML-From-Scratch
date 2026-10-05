"""Functional ops: gradient checks and parity with the hand-derived nn losses."""

from __future__ import annotations

import numpy as np

from scratchgrad.autograd import Tensor
from scratchgrad.autograd import functional as F
from scratchgrad.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)
Z = rng.standard_normal((5, 4)) * 3
Y = np.eye(4)[rng.integers(0, 4, 5)]


def check_scalar(f, x: np.ndarray) -> None:
    t = Tensor(x, requires_grad=True)
    f(t).backward()
    gradient_check(lambda x_: float(f(Tensor(x_)).data), t.grad, x)


class TestSoftmax:
    def test_rows_sum_to_one_and_match_reference(self) -> None:
        p = F.softmax(Tensor(Z)).data
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        e = np.exp(Z - Z.max(axis=1, keepdims=True))
        np.testing.assert_allclose(p, e / e.sum(axis=1, keepdims=True))

    def test_gradient(self) -> None:
        R = rng.standard_normal(Z.shape)
        check_scalar(lambda z: (F.softmax(z) * R).sum(), Z)

    def test_log_softmax_gradient(self) -> None:
        R = rng.standard_normal(Z.shape)
        check_scalar(lambda z: (F.log_softmax(z) * R).sum(), Z)

    def test_extreme_logits_stay_finite(self) -> None:
        out = F.log_softmax(Tensor([[1000.0, -1000.0, 0.0]])).data
        assert np.isfinite(out).all()


class TestLosses:
    def test_cross_entropy_gradient(self) -> None:
        check_scalar(lambda z: F.cross_entropy(z, Tensor(Y)), Z)

    def test_bce_gradient(self) -> None:
        z, y = rng.standard_normal((6, 1)), rng.integers(0, 2, (6, 1)) * 1.0
        check_scalar(lambda t: F.bce_with_logits(t, Tensor(y)), z)

    def test_mse_gradient(self) -> None:
        a, b = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
        check_scalar(lambda t: F.mse_loss(t, Tensor(b)), a)

    def test_bce_extreme_logits_stay_finite(self) -> None:
        z = Tensor([[800.0], [-800.0]], requires_grad=True)
        loss = F.bce_with_logits(z, Tensor([[1.0], [0.0]]))
        loss.backward()
        assert np.isfinite(loss.data) and np.isfinite(z.grad).all()


class TestParityWithNn:
    """The autograd gradient equals the hand-derived one (plan.md section 1)."""

    def test_cross_entropy(self) -> None:
        ref = CrossEntropyLoss()
        t = Tensor(Z, requires_grad=True)
        loss = F.cross_entropy(t, Tensor(Y))
        loss.backward()
        np.testing.assert_allclose(loss.data, ref.forward(Z, Y), atol=1e-10)
        np.testing.assert_allclose(t.grad, ref.backward(), atol=1e-10)

    def test_bce_with_logits(self) -> None:
        z, y = rng.standard_normal((6, 1)), rng.integers(0, 2, (6, 1)) * 1.0
        ref = BCEWithLogitsLoss()
        t = Tensor(z, requires_grad=True)
        loss = F.bce_with_logits(t, Tensor(y))
        loss.backward()
        np.testing.assert_allclose(loss.data, ref.forward(z, y), atol=1e-10)
        np.testing.assert_allclose(t.grad, ref.backward(), atol=1e-10)

    def test_mse(self) -> None:
        a, b = rng.standard_normal((4, 3)), rng.standard_normal((4, 3))
        ref = MSELoss()
        t = Tensor(a, requires_grad=True)
        loss = F.mse_loss(t, Tensor(b))
        loss.backward()
        np.testing.assert_allclose(loss.data, ref.forward(a, b), atol=1e-10)
        np.testing.assert_allclose(t.grad, ref.backward(), atol=1e-10)


class TestLayerNorm:
    X = rng.standard_normal((2, 3, 5)) * 2 + 1
    GAMMA = rng.standard_normal(5)
    BETA = rng.standard_normal(5)

    def test_matches_reference_formula(self) -> None:
        out = F.layer_norm(Tensor(self.X), Tensor(self.GAMMA), Tensor(self.BETA)).data
        mu = self.X.mean(axis=-1, keepdims=True)
        var = self.X.var(axis=-1, keepdims=True)
        ref = (self.X - mu) / np.sqrt(var + 1e-5) * self.GAMMA + self.BETA
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_unit_affine_gives_zero_mean_unit_variance(self) -> None:
        out = F.layer_norm(Tensor(self.X), Tensor(np.ones(5)), Tensor(np.zeros(5))).data
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-4)

    def test_input_gradient_matches_closed_form(self) -> None:
        # dx = (dxh - mean(dxh) - xh * mean(dxh * xh)) / std, dxh = dY * gamma
        R = rng.standard_normal(self.X.shape)
        x = Tensor(self.X, requires_grad=True)
        (F.layer_norm(x, Tensor(self.GAMMA), Tensor(self.BETA)) * R).sum().backward()
        mu = self.X.mean(axis=-1, keepdims=True)
        std = np.sqrt(self.X.var(axis=-1, keepdims=True) + 1e-5)
        xh = (self.X - mu) / std
        dxh = R * self.GAMMA
        expected = (
            dxh
            - dxh.mean(axis=-1, keepdims=True)
            - xh * (dxh * xh).mean(axis=-1, keepdims=True)
        ) / std
        np.testing.assert_allclose(x.grad, expected, atol=1e-10)

    def test_gradients_numerically(self) -> None:
        R = rng.standard_normal(self.X.shape)
        check_scalar(
            lambda x: (
                F.layer_norm(x, Tensor(self.GAMMA), Tensor(self.BETA)) * R
            ).sum(),
            self.X,
        )
        check_scalar(
            lambda g: (F.layer_norm(Tensor(self.X), g, Tensor(self.BETA)) * R).sum(),
            self.GAMMA,
        )
        check_scalar(
            lambda b: (F.layer_norm(Tensor(self.X), Tensor(self.GAMMA), b) * R).sum(),
            self.BETA,
        )


class TestGelu:
    def test_known_values(self) -> None:
        out = F.gelu(Tensor([-100.0, 0.0, 100.0])).data
        np.testing.assert_allclose(out, [0.0, 0.0, 100.0], atol=1e-9)
        np.testing.assert_allclose(
            F.gelu(Tensor([1.0])).data, [0.8411919906], atol=1e-9
        )

    def test_gradient(self) -> None:
        # Z / 3: strongly negative inputs have ~0 gradient, which breaks relative error
        R = rng.standard_normal(Z.shape)
        check_scalar(lambda z: (F.gelu(z) * R).sum(), Z / 3)
