r"""Two-dimensional convolution via im2col and matrix multiplication.

Full derivation: docs/derivations/conv_pool.md sections 1--3.
"""

from __future__ import annotations

import operator

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.validation import check_random_state

_WEIGHT_INITS = ("he", "xavier", "zeros")


def _pair(
    value: int | tuple[int, int], name: str, *, allow_zero: bool
) -> tuple[int, int]:
    """Normalize an integer or pair, validating both entries."""
    values = value if isinstance(value, tuple) else (value, value)
    if len(values) != 2:
        raise ValueError(f"{name} must be an int or a pair, got {value!r}")
    try:
        pair = (operator.index(values[0]), operator.index(values[1]))
    except TypeError as exc:
        raise ValueError(f"{name} must contain integers, got {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if any(entry < minimum for entry in pair):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} entries must be {relation}, got {value!r}")
    return pair


def _output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Return one convolution output dimension, rejecting an oversized kernel."""
    numerator = input_size + 2 * padding - kernel_size
    if numerator < 0:
        raise ValueError(
            "kernel_size cannot exceed the padded input size: "
            f"input={input_size}, kernel={kernel_size}, padding={padding}"
        )
    return numerator // stride + 1


def _im2col(
    x: FloatArray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[FloatArray, int, int]:
    """Unroll every NCHW receptive field into one row of a matrix."""
    if x.ndim != 4:
        raise ValueError(f"x must have shape (N, C, H, W), got {x.shape}")

    n, channels, height, width = x.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = _output_size(height, kernel_h, stride_h, pad_h)
    out_w = _output_size(width, kernel_w, stride_w, pad_w)

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    patches = np.empty((n, out_h, out_w, channels, kernel_h, kernel_w), dtype=x.dtype)
    for row in range(out_h):
        row_start = row * stride_h
        for col in range(out_w):
            col_start = col * stride_w
            patches[:, row, col] = x_padded[
                :,
                :,
                row_start : row_start + kernel_h,
                col_start : col_start + kernel_w,
            ]
    return patches.reshape(n * out_h * out_w, -1), out_h, out_w


def _col2im(
    columns: FloatArray,
    input_shape: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    out_h: int,
    out_w: int,
) -> FloatArray:
    """Scatter rows back into NCHW positions, summing overlapping patches."""
    n, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    patches = columns.reshape(n, out_h, out_w, channels, kernel_h, kernel_w)
    dx_padded = np.zeros(
        (n, channels, height + 2 * pad_h, width + 2 * pad_w),
        dtype=columns.dtype,
    )
    for row in range(out_h):
        row_start = row * stride_h
        for col in range(out_w):
            col_start = col * stride_w
            dx_padded[
                :,
                :,
                row_start : row_start + kernel_h,
                col_start : col_start + kernel_w,
            ] += patches[:, row, col]

    row_slice = slice(pad_h, pad_h + height)
    col_slice = slice(pad_w, pad_w + width)
    return dx_padded[:, :, row_slice, col_slice]


def _conv2d_forward(
    x: FloatArray,
    W: FloatArray,
    b: FloatArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> FloatArray:
    """Compute an NCHW cross-correlation by ``im2col @ W.T + b``."""
    if W.ndim != 4:
        raise ValueError(
            f"W must have shape (out_channels, in_channels, Kh, Kw), got {W.shape}"
        )
    out_channels, in_channels, kernel_h, kernel_w = W.shape
    if x.ndim != 4:
        raise ValueError(f"x must have shape (N, C, H, W), got {x.shape}")
    if x.shape[1] != in_channels:
        raise ValueError(f"x has {x.shape[1]} channels, but W expects {in_channels}")
    if b is not None and b.shape != (out_channels,):
        raise ValueError(f"b must have shape ({out_channels},), got {b.shape}")

    x_cols, out_h, out_w = _im2col(x, (kernel_h, kernel_w), stride, padding)
    W_cols = W.reshape(out_channels, -1)
    y_cols = x_cols @ W_cols.T
    if b is not None:
        y_cols += b
    n = x.shape[0]
    return y_cols.reshape(n, out_h, out_w, out_channels).transpose(0, 3, 1, 2)


def _conv2d_backward(
    x: FloatArray,
    W: FloatArray,
    grad_output: FloatArray,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute ``(dX, dW, db)`` for :func:`_conv2d_forward`."""
    out_channels, _, kernel_h, kernel_w = W.shape
    x_cols, out_h, out_w = _im2col(x, (kernel_h, kernel_w), stride, padding)
    expected_shape = (x.shape[0], out_channels, out_h, out_w)
    if grad_output.shape != expected_shape:
        raise ValueError(
            f"grad_output must have shape {expected_shape}, got {grad_output.shape}"
        )

    grad_cols = grad_output.transpose(0, 2, 3, 1).reshape(-1, out_channels)
    W_cols = W.reshape(out_channels, -1)
    dW = (grad_cols.T @ x_cols).reshape(W.shape)
    db = grad_cols.sum(axis=0)
    dx_cols = grad_cols @ W_cols
    dx = _col2im(
        dx_cols,
        x.shape,
        (kernel_h, kernel_w),
        stride,
        padding,
        out_h,
        out_w,
    )
    return dx, dW, db


class Conv2d(Module):
    r"""Two-dimensional NCHW cross-correlation implemented with im2col.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input. Must be positive.
    out_channels : int
        Number of filters/output channels. Must be positive.
    kernel_size : int or tuple of (int, int)
        Filter height and width. Entries must be positive.
    stride : int or tuple of (int, int), default=1
        Spatial step between adjacent receptive fields.
    padding : int or tuple of (int, int), default=0
        Symmetric zero-padding on both sides of each spatial dimension.
    bias : bool, default=True
        Whether to add a learnable per-output-channel bias.
    weight_init : {"he", "xavier", "zeros"}, default="he"
        Filter initialization. Convolutional fan-in is
        ``in_channels * kernel_height * kernel_width``; fan-out is the
        analogous quantity using ``out_channels``.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds filter initialization.

    Attributes
    ----------
    W : ndarray of shape (out_channels, in_channels, kernel_height, kernel_width)
    b : ndarray of shape (out_channels,) or None

    Raises
    ------
    ValueError
        If channel counts, kernel/stride entries, or initialization are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Conv2d(1, 4, kernel_size=3, padding=1, random_state=0)
    >>> layer.forward(np.ones((2, 1, 8, 8))).shape
    (2, 4, 8, 8)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        weight_init: str = "he",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if weight_init not in _WEIGHT_INITS:
            raise ValueError(
                f"weight_init must be one of {_WEIGHT_INITS}, got {weight_init!r}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size, "kernel_size", allow_zero=False)
        self.stride = _pair(stride, "stride", allow_zero=False)
        self.padding = _pair(padding, "padding", allow_zero=True)
        self.bias = bias
        self.weight_init = weight_init
        self.random_state = random_state

        kernel_h, kernel_w = self.kernel_size
        shape = (out_channels, in_channels, kernel_h, kernel_w)
        fan_in = in_channels * kernel_h * kernel_w
        fan_out = out_channels * kernel_h * kernel_w
        rng = check_random_state(random_state)
        if weight_init == "he":
            self.W = rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)
        elif weight_init == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = rng.uniform(-limit, limit, size=shape)
        else:
            self.W = np.zeros(shape, dtype=np.float64)
        self.W = self.W.astype(np.float64)
        self.b = np.zeros(out_channels, dtype=np.float64) if bias else None

        self._x: FloatArray | None = None
        self._dW: FloatArray | None = None
        self._db: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute the cross-correlation and cache ``x`` for backward."""
        self._x = x
        return _conv2d_forward(x, self.W, self.b, self.stride, self.padding)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute and cache filter/bias gradients, and return ``dX``."""
        dx, self._dW, db = _conv2d_backward(
            self._x, self.W, grad_output, self.stride, self.padding
        )
        self._db = db if self.bias else None
        return dx

    def parameters(self) -> list[FloatArray]:
        """Return ``[W, b]`` when biased, otherwise ``[W]``."""
        return [self.W, self.b] if self.bias else [self.W]

    def grads(self) -> list[FloatArray]:
        """Return gradients in the same order as :meth:`parameters`."""
        return [self._dW, self._db] if self.bias else [self._dW]
