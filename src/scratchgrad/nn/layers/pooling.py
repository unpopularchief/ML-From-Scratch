r"""Two-dimensional max pooling with an explicit scatter-add backward pass.

Full derivation: docs/derivations/conv_pool.md section 4.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.layers.conv2d import _output_size, _pair
from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray, IntArray


def _maxpool2d_forward(
    x: FloatArray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[FloatArray, IntArray]:
    """Return pooled NCHW output and each window's first argmax index."""
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
        constant_values=-np.inf,
    )
    y = np.empty((n, channels, out_h, out_w), dtype=x.dtype)
    argmax = np.empty((n, channels, out_h, out_w), dtype=np.int64)
    for row in range(out_h):
        row_start = row * stride_h
        for col in range(out_w):
            col_start = col * stride_w
            window = x_padded[
                :,
                :,
                row_start : row_start + kernel_h,
                col_start : col_start + kernel_w,
            ].reshape(n, channels, -1)
            argmax[:, :, row, col] = np.argmax(window, axis=-1)
            y[:, :, row, col] = np.max(window, axis=-1)
    return y, argmax


def _maxpool2d_backward(
    grad_output: FloatArray,
    argmax: IntArray,
    input_shape: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> FloatArray:
    """Route each output gradient to its cached winner, summing overlaps."""
    n, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = _output_size(height, kernel_h, stride_h, pad_h)
    out_w = _output_size(width, kernel_w, stride_w, pad_w)
    expected_shape = (n, channels, out_h, out_w)
    if grad_output.shape != expected_shape:
        raise ValueError(
            f"grad_output must have shape {expected_shape}, got {grad_output.shape}"
        )
    if argmax.shape != expected_shape:
        raise ValueError(f"argmax must have shape {expected_shape}, got {argmax.shape}")

    dx_padded = np.zeros(
        (n, channels, height + 2 * pad_h, width + 2 * pad_w),
        dtype=grad_output.dtype,
    )
    for row in range(out_h):
        row_start = row * stride_h
        for col in range(out_w):
            col_start = col * stride_w
            grad_window = np.zeros(
                (n, channels, kernel_h * kernel_w), dtype=grad_output.dtype
            )
            np.put_along_axis(
                grad_window,
                argmax[:, :, row, col, None],
                grad_output[:, :, row, col, None],
                axis=-1,
            )
            dx_padded[
                :,
                :,
                row_start : row_start + kernel_h,
                col_start : col_start + kernel_w,
            ] += grad_window.reshape(n, channels, kernel_h, kernel_w)

    return dx_padded[:, :, pad_h : pad_h + height, pad_w : pad_w + width]


class MaxPool2d(Module):
    """Maximum over sliding NCHW spatial windows.

    Parameters
    ----------
    kernel_size : int or tuple of (int, int)
        Pooling-window height and width. Entries must be positive.
    stride : int, tuple of (int, int), or None, default=None
        Spatial step. ``None`` uses ``kernel_size`` (non-overlapping windows).
    padding : int or tuple of (int, int), default=0
        Symmetric negative-infinity padding. Each entry may not exceed half
        the corresponding kernel size, matching PyTorch's contract.

    Notes
    -----
    At a tie, backward routes the gradient to the first maximum in row-major
    window order. This is the deterministic subgradient selected by
    ``numpy.argmax`` and ``torch.nn.MaxPool2d``.

    Raises
    ------
    ValueError
        If kernel/stride/padding entries or an input shape are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> layer = MaxPool2d(kernel_size=2)
    >>> layer.forward(np.arange(16.0).reshape(1, 1, 4, 4))
    array([[[[ 5.,  7.],
             [13., 15.]]]])

    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.kernel_size = _pair(kernel_size, "kernel_size", allow_zero=False)
        self.stride = (
            self.kernel_size
            if stride is None
            else _pair(stride, "stride", allow_zero=False)
        )
        self.padding = _pair(padding, "padding", allow_zero=True)
        for pad, kernel in zip(self.padding, self.kernel_size, strict=True):
            if pad > kernel // 2:
                raise ValueError(
                    "padding must be at most half the kernel size, got "
                    f"padding={self.padding}, kernel_size={self.kernel_size}"
                )

        self._input_shape: tuple[int, ...] | None = None
        self._argmax: IntArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Pool ``x`` and cache the winning position in every window."""
        self._input_shape = x.shape
        y, self._argmax = _maxpool2d_forward(
            x, self.kernel_size, self.stride, self.padding
        )
        return y

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Scatter output gradients to the cached maxima."""
        return _maxpool2d_backward(
            grad_output,
            self._argmax,
            self._input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
        )
