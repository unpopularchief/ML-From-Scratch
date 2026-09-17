"""PyTorch parity for scratchgrad.nn's Dropout and BatchNorm1d.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

``BatchNorm1d`` is exact parity in both modes (same published formulas,
same tier as ``optim``/``nn.md``'s other components). ``Dropout``'s RNG
stream differs from PyTorch's, so training mode is a statistical check
only; eval mode (a pure identity on both sides) is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import BatchNorm1d, Dropout

pytestmark = pytest.mark.reference


def test_batchnorm_train_mode_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 5))
    grad_out = rng.standard_normal((8, 5))

    ours = BatchNorm1d(5)
    ours.gamma = rng.standard_normal(5)
    ours.beta = rng.standard_normal(5)
    y_ours = ours.forward(X)
    dX_ours = ours.backward(grad_out)

    theirs = torch.nn.BatchNorm1d(5, dtype=torch.float64)
    with torch.no_grad():
        theirs.weight.copy_(torch.tensor(ours.gamma.copy()))
        theirs.bias.copy_(torch.tensor(ours.beta.copy()))
    theirs.train()
    x_t = torch.tensor(X, requires_grad=True)
    y_theirs = theirs(x_t)
    y_theirs.backward(torch.tensor(grad_out))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-8)
    np.testing.assert_allclose(dX_ours, x_t.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(ours.grads()[0], theirs.weight.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(ours.grads()[1], theirs.bias.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(
        ours.running_mean, theirs.running_mean.numpy(), atol=1e-8
    )
    np.testing.assert_allclose(ours.running_var, theirs.running_var.numpy(), atol=1e-8)


def test_batchnorm_eval_mode_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((8, 5))
    grad_out = rng.standard_normal((8, 5))

    ours = BatchNorm1d(5)
    ours.gamma = rng.standard_normal(5)
    ours.beta = rng.standard_normal(5)
    ours.running_mean = rng.standard_normal(5)
    ours.running_var = np.abs(rng.standard_normal(5)) + 0.5
    ours.eval()
    y_ours = ours.forward(X)
    dX_ours = ours.backward(grad_out)

    theirs = torch.nn.BatchNorm1d(5, dtype=torch.float64)
    with torch.no_grad():
        theirs.weight.copy_(torch.tensor(ours.gamma.copy()))
        theirs.bias.copy_(torch.tensor(ours.beta.copy()))
        theirs.running_mean.copy_(torch.tensor(ours.running_mean.copy()))
        theirs.running_var.copy_(torch.tensor(ours.running_var.copy()))
    theirs.eval()
    x_t = torch.tensor(X, requires_grad=True)
    y_theirs = theirs(x_t)
    y_theirs.backward(torch.tensor(grad_out))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-8)
    np.testing.assert_allclose(dX_ours, x_t.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(ours.grads()[0], theirs.weight.grad.numpy(), atol=1e-8)
    np.testing.assert_allclose(ours.grads()[1], theirs.bias.grad.numpy(), atol=1e-8)


def test_dropout_eval_mode_matches_torch_exactly() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 5))

    ours = Dropout(p=0.5, random_state=0).eval()
    y_ours = ours.forward(X)

    theirs = torch.nn.Dropout(p=0.5)
    theirs.eval()
    x_t = torch.tensor(X)
    y_theirs = theirs(x_t)

    np.testing.assert_array_equal(y_ours, y_theirs.numpy())


def test_dropout_train_mode_keep_rate_matches_torch_statistically() -> None:
    torch = pytest.importorskip("torch")
    p = 0.3
    X = np.ones((5000, 20))

    ours_keep_rate = np.mean(Dropout(p=p, random_state=0).forward(X) != 0.0)

    x_t = torch.ones(5000, 20, dtype=torch.float64)
    theirs_keep_rate = float(
        (torch.nn.functional.dropout(x_t, p=p, training=True) != 0).float().mean()
    )

    # different RNG streams -- both should land near the same 1-p keep rate
    assert ours_keep_rate == pytest.approx(1.0 - p, abs=0.01)
    assert theirs_keep_rate == pytest.approx(1.0 - p, abs=0.01)
