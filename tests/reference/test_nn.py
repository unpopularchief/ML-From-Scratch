"""PyTorch parity for scratchgrad.nn.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

**Exact parity, not a tolerance-only outcome comparison** (unlike
``LinearSVM``/``RandomForest``, same tier as ``optim/``): these are the
same published formulas, so feed identical inputs through both sides and
assert forward outputs and backward gradients match tightly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import BCEWithLogitsLoss, CrossEntropyLoss, Linear, MSELoss
from scratchgrad.nn.activations import ReLU, Sigmoid, Softmax, Tanh

pytestmark = pytest.mark.reference


def test_linear_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4))
    grad_out = rng.standard_normal((5, 3))

    ours = Linear(4, 3, random_state=0)
    y_ours = ours.forward(X)
    dX_ours = ours.backward(grad_out)

    theirs = torch.nn.Linear(4, 3, dtype=torch.float64)
    with torch.no_grad():
        theirs.weight.copy_(torch.tensor(ours.W.T.copy()))  # torch: (out, in)
        theirs.bias.copy_(torch.tensor(ours.b.copy()))
    x_t = torch.tensor(X, requires_grad=True)
    y_theirs = theirs(x_t)
    y_theirs.backward(torch.tensor(grad_out))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dX_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(
        ours.grads()[0], theirs.weight.grad.numpy().T, atol=1e-10
    )
    np.testing.assert_allclose(ours.grads()[1], theirs.bias.grad.numpy(), atol=1e-10)


def _check_activation(ours_module, torch_fn) -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 5))
    grad_out = rng.standard_normal((6, 5))

    y_ours = ours_module.forward(x)
    dx_ours = ours_module.backward(grad_out)

    x_t = torch.tensor(x, requires_grad=True)
    y_theirs = torch_fn(x_t)
    y_theirs.backward(torch.tensor(grad_out))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)


def test_relu_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    _check_activation(ReLU(), torch.relu)


def test_sigmoid_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    _check_activation(Sigmoid(), torch.sigmoid)


def test_tanh_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    _check_activation(Tanh(), torch.tanh)


def test_softmax_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    _check_activation(Softmax(), lambda z: torch.softmax(z, dim=-1))


def test_mse_loss_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    y_pred = rng.standard_normal((6, 3))
    y_true = rng.standard_normal((6, 3))

    loss = MSELoss()
    value_ours = loss.forward(y_pred, y_true)
    grad_ours = loss.backward()

    pred_t = torch.tensor(y_pred, requires_grad=True)
    value_theirs = torch.nn.functional.mse_loss(pred_t, torch.tensor(y_true))
    value_theirs.backward()

    assert value_ours == pytest.approx(value_theirs.item(), abs=1e-10)
    np.testing.assert_allclose(grad_ours, pred_t.grad.numpy(), atol=1e-10)


def test_bce_with_logits_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    z = rng.standard_normal(8)
    y = rng.integers(0, 2, size=8).astype(np.float64)

    loss = BCEWithLogitsLoss()
    value_ours = loss.forward(z, y)
    grad_ours = loss.backward()

    z_t = torch.tensor(z, requires_grad=True)
    value_theirs = torch.nn.functional.binary_cross_entropy_with_logits(
        z_t, torch.tensor(y)
    )
    value_theirs.backward()

    assert value_ours == pytest.approx(value_theirs.item(), abs=1e-10)
    np.testing.assert_allclose(grad_ours, z_t.grad.numpy(), atol=1e-10)


def test_cross_entropy_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((6, 4))
    labels = rng.integers(0, 4, size=6)
    y_true = np.eye(4)[labels]

    loss = CrossEntropyLoss()
    value_ours = loss.forward(Z, y_true)
    grad_ours = loss.backward()

    Z_t = torch.tensor(Z, requires_grad=True)
    value_theirs = torch.nn.functional.cross_entropy(
        Z_t, torch.tensor(labels, dtype=torch.long)
    )
    value_theirs.backward()

    assert value_ours == pytest.approx(value_theirs.item(), abs=1e-10)
    np.testing.assert_allclose(grad_ours, Z_t.grad.numpy(), atol=1e-10)
