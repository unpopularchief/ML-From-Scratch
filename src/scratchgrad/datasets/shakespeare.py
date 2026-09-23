"""tiny-Shakespeare loader: download once, then reuse a local text cache."""

from __future__ import annotations

import urllib.request
from pathlib import Path

_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tiny_shakespeare(
    data_path: str | Path = ".cache/tiny-shakespeare/input.txt",
    download: bool = True,
) -> str:
    """Load tiny Shakespeare, downloading it into ``data_path`` when absent.

    The corpus is kept out of version control like MNIST. Set ``download`` to
    ``False`` to require an existing local cache without making a network call.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        if not download:
            raise FileNotFoundError(f"{data_path} not found and download=False.")
        data_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        urllib.request.urlretrieve(_URL, data_path)  # pragma: no cover
    return data_path.read_text(encoding="utf-8")
