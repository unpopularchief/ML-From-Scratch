"""Tests for scratchgrad.datasets.mnist's IDX byte-parsing.

No network access here -- only ``_parse_idx_images``/``_parse_idx_labels``
are tested, against small hand-built IDX byte blobs, plus
``load_mnist``'s ``download=False`` guard (deterministic, no network),
per handoff.md's scoping decision for this unit: the actual download+cache
path is exercised only by manually running ``examples/mnist.py``, not by
the automated test suite.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from scratchgrad.datasets.mnist import _parse_idx_images, _parse_idx_labels, load_mnist


def _build_idx_images(pixels: np.ndarray) -> bytes:
    """Pack a (n, rows, cols) uint8 array into IDX3 bytes (magic=2051)."""
    n, rows, cols = pixels.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    return header + pixels.astype(np.uint8).tobytes()


def _build_idx_labels(labels: np.ndarray) -> bytes:
    """Pack a (n,) uint8 array into IDX1 bytes (magic=2049)."""
    header = struct.pack(">II", 2049, len(labels))
    return header + labels.astype(np.uint8).tobytes()


class TestParseIdxImages:
    def test_shape_and_scaling(self) -> None:
        pixels = np.array([[[0, 128], [255, 64]], [[10, 20], [30, 40]]], dtype=np.uint8)
        raw = _build_idx_images(pixels)

        X = _parse_idx_images(raw)

        assert X.shape == (2, 4)
        np.testing.assert_allclose(X[0], [0.0, 128 / 255, 255 / 255, 64 / 255])

    def test_dtype_is_float64_in_zero_one(self) -> None:
        pixels = np.full((3, 2, 2), 255, dtype=np.uint8)
        X = _parse_idx_images(_build_idx_images(pixels))
        assert X.dtype == np.float64
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_wrong_magic_raises(self) -> None:
        bad = struct.pack(">IIII", 9999, 1, 2, 2) + bytes(4)
        with pytest.raises(ValueError, match="magic"):
            _parse_idx_images(bad)


class TestParseIdxLabels:
    def test_values_and_dtype(self) -> None:
        labels = np.array([0, 9, 3, 7], dtype=np.uint8)
        y = _parse_idx_labels(_build_idx_labels(labels))
        np.testing.assert_array_equal(y, [0, 9, 3, 7])
        assert y.dtype == np.int64

    def test_wrong_magic_raises(self) -> None:
        bad = struct.pack(">II", 1234, 1) + bytes(1)
        with pytest.raises(ValueError, match="magic"):
            _parse_idx_labels(bad)


class TestLoadMnistDownloadGuard:
    def test_download_false_with_missing_files_raises_without_network(
        self, tmp_path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="download=False"):
            load_mnist(data_dir=tmp_path, download=False)

    def test_download_false_skips_an_already_cached_file(self, tmp_path) -> None:
        # train-images-idx3-ubyte.gz already "cached" (dummy content -- never
        # parsed, since the next file is still missing and raises first) --
        # exercises the `if path.exists(): continue` branch without network.
        (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="train-labels"):
            load_mnist(data_dir=tmp_path, download=False)
