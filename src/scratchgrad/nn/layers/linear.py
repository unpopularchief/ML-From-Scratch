r"""Linear (fully-connected / affine) layer: :math:`Y = XW + b`.

Full derivation: docs/derivations/nn.md section 2.
"""

from __future__ import annotations

from scratchgrad.nn.init import he_normal, xavier_uniform, zeros
from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.validation import check_random_state

_WEIGHT_INITS = ("he", "xavier", "zeros")


def _linear_forward(X: FloatArray, W: FloatArray, b: FloatArray) -> FloatArray:
    """Compute ``Y = X @ W + b``."""
    return X @ W + b


def _linear_backward(
    X: FloatArray, W: FloatArray, grad_output: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute ``(dX, dW, db)`` from ``grad_output = dL/dY``."""
    dW = X.T @ grad_output  # dL/dW = X^T @ dL/dY
    db = grad_output.sum(axis=0)  # dL/db = column sum of dL/dY over samples
    dX = grad_output @ W.T  # dL/dX = dL/dY @ W^T
    return dX, dW, db


class Linear(Module):
    r"""Fully-connected layer, :math:`Y = XW + b`.

    Parameters
    ----------
    in_features : int
        Input dimension :math:`d_{in}`. Must be positive.
    out_features : int
        Output dimension :math:`d_{out}`. Must be positive.
    weight_init : {"he", "xavier", "zeros"}, default="he"
        Scheme used to initialize ``W`` (see ``scratchgrad.nn.init``);
        ``b`` always starts at zero.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds ``W``'s initialization.

    Attributes
    ----------
    W : ndarray of shape (in_features, out_features)
    b : ndarray of shape (out_features,)

    Raises
    ------
    ValueError
        If ``in_features``/``out_features`` is not positive, or
        ``weight_init`` is not one of the options above.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Linear(3, 2, random_state=0)
    >>> x = np.ones((4, 3))
    >>> y = layer.forward(x)
    >>> y.shape
    (4, 2)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: str = "he",
        random_state: int | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if weight_init not in _WEIGHT_INITS:
            raise ValueError(
                f"weight_init must be one of {_WEIGHT_INITS}, got {weight_init!r}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.random_state = random_state

        rng = check_random_state(random_state)
        if weight_init == "he":
            self.W = he_normal(in_features, out_features, rng)
        elif weight_init == "xavier":
            self.W = xavier_uniform(in_features, out_features, rng)
        else:
            self.W = zeros((in_features, out_features))
        self.b = zeros((out_features,))

        self._x: FloatArray | None = None
        self._dW: FloatArray | None = None
        self._db: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``Y = X @ W + b``, caching ``x`` for :meth:`backward`."""
        self._x = x
        return _linear_forward(x, self.W, self.b)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute and cache ``(dW, db)``, and return ``dX``."""
        dX, self._dW, self._db = _linear_backward(self._x, self.W, grad_output)
        return dX

    def parameters(self) -> list[FloatArray]:
        """``[W, b]``."""
        return [self.W, self.b]

    def grads(self) -> list[FloatArray]:
        """``[dW, db]`` from the most recent :meth:`backward` call."""
        return [self._dW, self._db]
