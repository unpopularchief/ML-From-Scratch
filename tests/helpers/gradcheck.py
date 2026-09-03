"""Gradient checking via central finite differences.

The single most important test helper in this project (see plan.md
section 3): every hand-derived ``backward()`` written from M3 onward is
checked against this, not against "it looks right". If the analytic and
numerical gradients agree, the derivation was implemented correctly,
independent of whether the resulting model happens to train well.

Not used in M0 itself (there are no gradients yet), but written now so
gradient checking is a solved problem before the first one is needed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def numerical_gradient(
    f: Callable[[FloatArray], float],
    x: FloatArray,
    eps: float = 1e-6,
) -> FloatArray:
    r"""Estimate :math:`\nabla f(x)` via central finite differences.

    For each coordinate :math:`x_i`:

    .. math::
        \frac{\partial f}{\partial x_i} \approx
            \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2 \epsilon}

    The central (two-sided) difference is used rather than the forward
    difference :math:`(f(x+\epsilon) - f(x)) / \epsilon` because its
    error is :math:`O(\epsilon^2)` instead of :math:`O(\epsilon)` — it
    cancels the first-order term of the Taylor expansion that the forward
    difference leaves in, which is what makes a loose ``rtol`` in
    :func:`gradient_check` viable at all.

    Parameters
    ----------
    f : callable
        Maps an ndarray of the same shape as ``x`` to a scalar float.
    x : ndarray
        Point to differentiate at, any shape.
    eps : float, default=1e-6
        Step size. Small enough that the :math:`O(\epsilon^2)` truncation
        error is negligible, large enough to stay above float64 rounding
        noise — 1e-6 sits in the middle of that trade-off for typical
        loss-function magnitudes.

    Returns
    -------
    ndarray of the same shape as ``x``
        Estimated gradient.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    # np.nditer walks every coordinate regardless of x's shape (vector,
    # weight matrix, ...) without the caller needing to flatten it first.
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        index = it.multi_index
        original_value = x[index]

        x[index] = original_value + eps
        f_plus = f(x)

        x[index] = original_value - eps
        f_minus = f(x)

        x[index] = original_value  # restore before moving to the next coordinate
        grad[index] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def relative_error(analytic: FloatArray, numerical: FloatArray) -> float:
    r"""Elementwise-max relative error between two gradients.

    .. math::
        \max_i \frac{|a_i - n_i|}{\max(|a_i|, |n_i|, \epsilon_{\text{floor}})}

    A relative (rather than absolute) error is used because gradients
    can legitimately span many orders of magnitude between parameters;
    ``eps_floor`` keeps the denominator from vanishing when both
    gradients are near zero, where absolute noise dominates and the
    relative error would otherwise be a meaningless division by ~0.

    Parameters
    ----------
    analytic : ndarray
        Gradient computed by the ``backward()`` implementation under test.
    numerical : ndarray
        Gradient from :func:`numerical_gradient`, same shape as ``analytic``.

    Returns
    -------
    float
    """
    eps_floor = 1e-12
    numerator = np.abs(analytic - numerical)
    denominator = np.maximum(np.abs(analytic), np.abs(numerical))
    denominator = np.maximum(denominator, eps_floor)
    return float(np.max(numerator / denominator))


def gradient_check(
    f: Callable[[FloatArray], float],
    analytic_grad: FloatArray,
    x: FloatArray,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> None:
    """Assert that ``analytic_grad`` matches the numerical gradient of ``f`` at ``x``.

    This is the function every backward-pass test calls. It exists so
    each test site reads as one line stating *what* is being checked,
    with the finite-difference mechanics factored out here.

    Parameters
    ----------
    f : callable
        Scalar-valued function (e.g. a loss) that ``analytic_grad`` is
        claimed to be the gradient of at ``x``.
    analytic_grad : ndarray, same shape as ``x``
        Gradient produced by the implementation under test.
    x : ndarray
        Point to check the gradient at.
    eps : float, default=1e-6
        Passed to :func:`numerical_gradient`.
    tol : float, default=1e-4
        Maximum acceptable :func:`relative_error`. 1e-4 comfortably
        separates a genuinely wrong gradient (typically off by orders of
        magnitude, a sign error, or a missing term) from the ~1e-6-1e-8
        floating-point noise finite differences carry even when correct.

    Raises
    ------
    AssertionError
        If the relative error between ``analytic_grad`` and the numerical
        gradient exceeds ``tol``.
    """
    numerical_grad = numerical_gradient(
        f, np.array(x, dtype=np.float64, copy=True), eps=eps
    )
    error = relative_error(np.asarray(analytic_grad, dtype=np.float64), numerical_grad)
    assert error <= tol, (
        f"Gradient check failed: relative error {error:.2e} exceeds tol={tol:.2e}.\n"
        f"analytic:\n{analytic_grad}\nnumerical:\n{numerical_grad}"
    )
