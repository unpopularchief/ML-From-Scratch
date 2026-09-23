"""Tests for the tiny-Shakespeare cache loader, without network access."""

from __future__ import annotations

import pytest

from scratchgrad.datasets.shakespeare import load_tiny_shakespeare


def test_loads_an_existing_utf8_cache(tmp_path) -> None:
    corpus = "To be, or not to be.\n"
    data_path = tmp_path / "input.txt"
    data_path.write_text(corpus, encoding="utf-8")

    assert load_tiny_shakespeare(data_path, download=False) == corpus


def test_download_false_with_missing_cache_raises_without_network(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download=False"):
        load_tiny_shakespeare(tmp_path / "input.txt", download=False)
