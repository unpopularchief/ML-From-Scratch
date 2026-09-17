"""MNIST loader: downloads into a gitignored cache dir, parses IDX by hand.

Per ``generators.py``'s own docstring and `plan.md` section 8 mistake 10
("committing datasets"): real datasets are never checked into the repo,
only cached on first use. The IDX file format itself (magic number, a
dimension header, then raw bytes) is parsed here by hand with ``struct``/
``numpy`` rather than via a dataset library -- this project's usual
"understand the format, don't just call a loader" stance, applied to data
loading instead of a model.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from scratchgrad.typing import FloatArray, IntArray

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def _parse_idx_images(raw: bytes) -> FloatArray:
    """Parse an IDX3 image file: magic(4) + n(4) + rows(4) + cols(4) + pixels.

    Scales pixels from ``[0, 255]`` to ``[0, 1]`` -- the dtype-policy
    boundary (`docs/conventions.md`) where raw ``uint8`` bytes become this
    project's one float64 convention.
    """
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != _IMAGE_MAGIC:
        raise ValueError(
            f"Not an IDX3 image file (magic={magic}, expected {_IMAGE_MAGIC})."
        )
    pixels = np.frombuffer(raw[16:], dtype=np.uint8)
    return pixels.reshape(n, rows * cols).astype(np.float64) / 255.0


def _parse_idx_labels(raw: bytes) -> IntArray:
    """Parse an IDX1 label file: magic(4) + n(4) + one byte per label."""
    magic, n = struct.unpack(">II", raw[:8])
    if magic != _LABEL_MAGIC:
        raise ValueError(
            f"Not an IDX1 label file (magic={magic}, expected {_LABEL_MAGIC})."
        )
    return np.frombuffer(raw[8 : 8 + n], dtype=np.uint8).astype(np.int64)


def load_mnist(
    data_dir: str | Path = ".cache/mnist", download: bool = True
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """Load MNIST, downloading any missing file into ``data_dir`` first.

    Parameters
    ----------
    data_dir : str or Path, default=".cache/mnist"
        Directory the four gzipped IDX files are cached in (created if
        missing). Never committed -- see ``.gitignore``'s ``.cache/``.
    download : bool, default=True
        If ``False``, a missing file raises ``FileNotFoundError`` instead
        of reaching out to the network.

    Returns
    -------
    X_train : ndarray of shape (60000, 784), float64 in [0, 1]
    y_train : ndarray of shape (60000,), int64 in [0, 9]
    X_test : ndarray of shape (10000, 784), float64 in [0, 1]
    y_test : ndarray of shape (10000,), int64 in [0, 9]

    Raises
    ------
    FileNotFoundError
        If ``download=False`` and a required file is missing.

    """
    data_dir = Path(data_dir)
    paths = {name: data_dir / filename for name, filename in _FILES.items()}

    for name, path in paths.items():
        if path.exists():
            continue
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False.")
        path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        urllib.request.urlretrieve(_MIRROR + _FILES[name], path)  # pragma: no cover

    # Network/filesystem I/O -- not unit-tested (see tests/datasets/test_mnist.py's
    # module docstring); _parse_idx_images/_parse_idx_labels are tested directly
    # against hand-built byte blobs instead.
    with gzip.open(paths["train_images"], "rb") as f:  # pragma: no cover
        X_train = _parse_idx_images(f.read())
    with gzip.open(paths["train_labels"], "rb") as f:  # pragma: no cover
        y_train = _parse_idx_labels(f.read())
    with gzip.open(paths["test_images"], "rb") as f:  # pragma: no cover
        X_test = _parse_idx_images(f.read())
    with gzip.open(paths["test_labels"], "rb") as f:  # pragma: no cover
        y_test = _parse_idx_labels(f.read())

    return X_train, y_train, X_test, y_test  # pragma: no cover
