r"""BatchNorm1d: normalize each feature over the batch, then scale/shift.

Full derivation: docs/derivations/dropout_batchnorm.md section 3.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray


def _batchnorm_forward_train(
    x: FloatArray, gamma: FloatArray, beta: FloatArray, eps: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Normalize ``x`` by its own batch mean/var; return ``(y, x_hat, std, var)``."""
    mu = x.mean(axis=0)
    xmu = x - mu
    var = np.mean(xmu**2, axis=0)  # biased (ddof=0) -- the training-time denominator
    std = np.sqrt(var + eps)
    x_hat = xmu / std
    y = gamma * x_hat + beta
    return y, x_hat, std, var


def _batchnorm_forward_eval(
    x: FloatArray,
    gamma: FloatArray,
    beta: FloatArray,
    running_mean: FloatArray,
    running_var: FloatArray,
    eps: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Normalize ``x`` by the running mean/var; return ``(y, x_hat, std)``."""
    std = np.sqrt(running_var + eps)
    x_hat = (x - running_mean) / std
    y = gamma * x_hat + beta
    return y, x_hat, std


def _batchnorm_backward_train(
    x_hat: FloatArray, std: FloatArray, gamma: FloatArray, grad_output: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute ``(dX, dgamma, dbeta)`` when ``mu``/``var`` came from this batch.

    ``dX = (gamma/std) * (dx_hat - mean(dx_hat) - x_hat*mean(dx_hat*x_hat))``,
    the closed form for backpropagating through mean/var's dependence on
    *every* sample in the batch -- see docs/derivations/dropout_batchnorm.md
    section 3 for the full chain-rule derivation.
    """
    dgamma = np.sum(grad_output * x_hat, axis=0)  # dL/dgamma
    dbeta = np.sum(grad_output, axis=0)  # dL/dbeta
    dx_hat = grad_output * gamma  # dL/dx_hat
    dx = (
        dx_hat - dx_hat.mean(axis=0) - x_hat * (dx_hat * x_hat).mean(axis=0)
    ) / std  # dL/dX
    return dx, dgamma, dbeta


def _batchnorm_backward_eval(
    x_hat: FloatArray, std: FloatArray, gamma: FloatArray, grad_output: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute ``(dX, dgamma, dbeta)`` when ``mu``/``var`` are fixed running stats.

    Simpler than the training case: ``x_hat`` is then a per-feature affine
    function of ``X`` alone (no coupling across samples through a
    batch-dependent mean/var), so ``dX = grad_output * gamma / std`` directly.
    """
    dgamma = np.sum(grad_output * x_hat, axis=0)
    dbeta = np.sum(grad_output, axis=0)
    dx = grad_output * gamma / std
    return dx, dgamma, dbeta


class BatchNorm1d(Module):
    r"""Batch normalization over ``(n, num_features)`` input (Ioffe & Szegedy, 2015).

    Training: normalizes by the *current batch's* mean/variance, then
    updates ``running_mean``/``running_var`` (exponential moving averages,
    used at evaluation time). Evaluation: normalizes by the running
    statistics instead, with no further update to them.

    Parameters
    ----------
    num_features : int
        Number of features ``d``. Must be positive.
    eps : float, default=1e-5
        Added to the variance before the square root, for numerical
        stability. Matches ``torch.nn.BatchNorm1d``'s default.
    momentum : float, default=0.1
        Exponential-moving-average weight for the running statistics:
        ``running = (1-momentum)*running + momentum*batch_stat``. Matches
        ``torch.nn.BatchNorm1d``'s default and convention (the *new*
        batch statistic gets weight ``momentum``, not ``1-momentum``).

    Attributes
    ----------
    gamma : ndarray of shape (num_features,)
        Learnable scale, initialized to ones.
    beta : ndarray of shape (num_features,)
        Learnable shift, initialized to zeros.
    running_mean : ndarray of shape (num_features,)
        Not learnable -- updated in training-mode :meth:`forward`, read
        in evaluation-mode :meth:`forward`. Initialized to zeros.
    running_var : ndarray of shape (num_features,)
        Same as ``running_mean``, but tracks the **Bessel-corrected
        (unbiased)** variance (``var * n/(n-1)``) even though the batch
        itself is normalized with the *biased* variance -- this matches
        ``torch.nn.BatchNorm1d`` exactly (see the derivation doc). Initialized
        to ones.

    Raises
    ------
    ValueError
        If ``num_features`` is not positive, or :meth:`forward` is called
        in training mode with a batch size of 1 (the batch variance is
        undefined/degenerate there -- matches PyTorch's own error for this
        case).

    Examples
    --------
    >>> import numpy as np
    >>> bn = BatchNorm1d(3)
    >>> x = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    >>> y = bn.forward(x)
    >>> np.round(y.mean(axis=0), 8)
    array([0., 0., 0.])

    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        """See the class docstring for parameter descriptions."""
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self._x_hat: FloatArray | None = None
        self._std: FloatArray | None = None
        self._was_training: bool = True
        self._dgamma: FloatArray | None = None
        self._dbeta: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Normalize by batch stats (training) or running stats (eval)."""
        self._was_training = self.training
        if self.training:
            n = x.shape[0]
            if n <= 1:
                raise ValueError(
                    "BatchNorm1d requires a batch size > 1 in training mode "
                    f"(the batch variance is undefined for n={n})."
                )
            y, self._x_hat, self._std, var = _batchnorm_forward_train(
                x, self.gamma, self.beta, self.eps
            )
            mu = x.mean(axis=0)
            unbiased_var = var * n / (n - 1)  # Bessel-corrected, see class docstring
            self.running_mean = (1 - self.momentum) * self.running_mean + (
                self.momentum * mu
            )
            self.running_var = (1 - self.momentum) * self.running_var + (
                self.momentum * unbiased_var
            )
            return y
        y, self._x_hat, self._std = _batchnorm_forward_eval(
            x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps
        )
        return y

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dX``, dispatching on the mode :meth:`forward` ran in."""
        if self._was_training:
            dx, self._dgamma, self._dbeta = _batchnorm_backward_train(
                self._x_hat, self._std, self.gamma, grad_output
            )
        else:
            dx, self._dgamma, self._dbeta = _batchnorm_backward_eval(
                self._x_hat, self._std, self.gamma, grad_output
            )
        return dx

    def parameters(self) -> list[FloatArray]:
        """``[gamma, beta]``."""
        return [self.gamma, self.beta]

    def grads(self) -> list[FloatArray]:
        """``[dgamma, dbeta]`` from the most recent :meth:`backward` call."""
        return [self._dgamma, self._dbeta]
