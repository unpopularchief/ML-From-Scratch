"""PyTorch parity for scratchgrad.optim.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
PyTorch joins the ``reference`` extra at M3 (plan.md section 7) -- this is
the first module that needs it.

**Exact parity, not a tolerance-only outcome comparison** (unlike
``LinearSVM``/``RandomForest``): these are the same published update
formulas (docs/derivations/optim.md sections 2-6), so run identical
gradient sequences through both sides (each recomputed from its own
current parameter value against a shared toy quadratic, so any formula
mismatch would compound rather than cancel) and assert the trajectories
match tightly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD, Adam, Momentum, Nesterov, RMSprop

pytestmark = pytest.mark.reference

_A = np.diag([1.0, 5.0, 20.0])  # a fixed, mildly ill-conditioned quadratic
_THETA0 = np.array([3.0, -2.0, 1.0])
_N_STEPS = 50


def _run_ours(opt, n_steps=_N_STEPS):
    theta = _THETA0.copy()
    for _ in range(n_steps):
        opt.step([theta], [_A @ theta])
    return theta


def _run_torch(torch, make_optimizer, n_steps=_N_STEPS):
    param = torch.tensor(_THETA0.copy(), dtype=torch.float64, requires_grad=True)
    optimizer = make_optimizer([param])
    a = torch.tensor(_A, dtype=torch.float64)
    for _ in range(n_steps):
        optimizer.zero_grad()
        param.grad = a @ param.detach()
        optimizer.step()
    return param.detach().numpy()


def test_sgd_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    lr = 0.01

    ours = _run_ours(SGD(lr=lr))
    theirs = _run_torch(torch, lambda p: torch.optim.SGD(p, lr=lr))

    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_momentum_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    lr, momentum = 0.01, 0.9

    ours = _run_ours(Momentum(lr=lr, momentum=momentum))
    theirs = _run_torch(torch, lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum))

    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_nesterov_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    lr, momentum = 0.01, 0.9

    ours = _run_ours(Nesterov(lr=lr, momentum=momentum))
    theirs = _run_torch(
        torch,
        lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, nesterov=True),
    )

    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_rmsprop_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    lr, beta, eps = 0.05, 0.9, 1e-8

    ours = _run_ours(RMSprop(lr=lr, beta=beta, eps=eps))
    theirs = _run_torch(
        torch, lambda p: torch.optim.RMSprop(p, lr=lr, alpha=beta, eps=eps)
    )

    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_adam_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8

    ours = _run_ours(Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps))
    theirs = _run_torch(
        torch,
        lambda p: torch.optim.Adam(p, lr=lr, betas=(beta1, beta2), eps=eps),
    )

    np.testing.assert_allclose(ours, theirs, atol=1e-8)
