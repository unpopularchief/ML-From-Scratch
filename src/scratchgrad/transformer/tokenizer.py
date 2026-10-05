"""Character and greedy byte-pair tokenizers for small language models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def _check_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")


def _replace_pair(tokens: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    """Replace non-overlapping occurrences of ``pair`` from left to right."""
    result: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def _encode_tokens(
    tokens: list[str], token_to_id: dict[str, int], known_chars: set[str]
) -> list[int]:
    unknown = sorted(set(tokens).difference(known_chars))
    if unknown:
        raise ValueError(f"text contains characters not seen during fit: {unknown!r}")
    return [token_to_id[token] for token in tokens]


def _decode_tokens(ids: Iterable[int], vocabulary: tuple[str, ...]) -> str:
    pieces: list[str] = []
    for token_id in ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(f"token IDs must be integers, got {token_id!r}")
        if token_id < 0 or token_id >= len(vocabulary):
            raise ValueError(f"token ID out of range: {token_id}")
        pieces.append(vocabulary[token_id])
    return "".join(pieces)


class CharTokenizer:
    """Map each Unicode code point to a deterministic integer ID.

    The vocabulary is sorted by code point. Python iterates combining marks as
    separate code points; this tokenizer does not normalize Unicode text.

    Examples
    --------
    >>> tok = CharTokenizer().fit("cab")
    >>> tok.encode("cab")
    [2, 0, 1]
    >>> tok.decode([2, 0, 1])
    'cab'

    """

    def __init__(self) -> None:
        """Create an unfitted character tokenizer."""
        self.vocabulary: tuple[str, ...] = ()
        self._token_to_id: dict[str, int] = {}
        self._fitted = False

    def fit(self, text: str) -> CharTokenizer:
        """Build a sorted vocabulary from ``text`` and return this tokenizer."""
        _check_text(text)
        self.vocabulary = tuple(sorted(set(text)))
        self._token_to_id = {
            token: token_id for token_id, token in enumerate(self.vocabulary)
        }
        self._fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters seen during fitting."""
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        """Convert text to one integer ID per code point."""
        _check_text(text)
        if not self._fitted:
            raise ValueError("fit must be called before encode")
        return _encode_tokens(list(text), self._token_to_id, set(self.vocabulary))

    def decode(self, ids: Iterable[int]) -> str:
        """Convert character IDs back to text."""
        if not self._fitted:
            raise ValueError("fit must be called before decode")
        return _decode_tokens(ids, self.vocabulary)


class BPETokenizer:
    """Greedy pair-frequency BPE over Unicode code points.

    Training repeatedly merges the most frequent adjacent pair. Frequency ties
    are broken lexicographically by the pair of token strings. No pre-tokenizer
    is used, so merges may include spaces or span word boundaries.

    Examples
    --------
    >>> tok = BPETokenizer(max_merges=2).fit("abab")
    >>> tok.merges
    [('a', 'b'), ('ab', 'ab')]
    >>> tok.decode(tok.encode("abab"))
    'abab'

    """

    def __init__(self, max_merges: int = 100) -> None:
        """Create a BPE tokenizer with an upper bound on learned merges."""
        if not isinstance(max_merges, int) or isinstance(max_merges, bool):
            raise TypeError(f"max_merges must be an integer, got {max_merges!r}")
        if max_merges < 0:
            raise ValueError(f"max_merges must be non-negative, got {max_merges}")
        self.max_merges = max_merges
        self.merges: list[tuple[str, str]] = []
        self.vocabulary: tuple[str, ...] = ()
        self._token_to_id: dict[str, int] = {}
        self._characters: set[str] = set()
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        """Number of distinct character and merged tokens."""
        return len(self.vocabulary)

    def fit(self, text: str) -> BPETokenizer:
        """Learn up to ``max_merges`` pair rules from ``text``."""
        _check_text(text)
        self._characters = set(text)
        tokens = list(text)
        self.merges = []
        vocabulary = list(sorted(self._characters))

        for _ in range(self.max_merges):
            counts = Counter(zip(tokens, tokens[1:], strict=False))
            if not counts:
                break
            pair, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
            merged = "".join(pair)
            self.merges.append(pair)
            tokens = _replace_pair(tokens, pair, merged)
            if merged not in vocabulary:
                vocabulary.append(merged)

        self.vocabulary = tuple(vocabulary)
        self._token_to_id = {
            token: token_id for token_id, token in enumerate(self.vocabulary)
        }
        self._fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        """Apply learned merge rules in rank order and return token IDs."""
        _check_text(text)
        if not self._fitted:
            raise ValueError("fit must be called before encode")
        tokens = list(text)
        _encode_tokens(tokens, self._token_to_id, self._characters)
        for pair in self.merges:
            tokens = _replace_pair(tokens, pair, "".join(pair))
        return [self._token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        """Convert character or merged-token IDs back to text."""
        if not self._fitted:
            raise ValueError("fit must be called before decode")
        return _decode_tokens(ids, self.vocabulary)
