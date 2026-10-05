"""Character and BPE tokenizer contracts."""

import pytest

from scratchgrad.transformer import BPETokenizer, CharTokenizer


class TestCharTokenizer:
    def test_sorted_vocabulary_and_round_trip(self) -> None:
        tokenizer = CharTokenizer().fit("cabac")

        assert tokenizer.vocabulary == ("a", "b", "c")
        assert tokenizer.vocab_size == 3
        assert tokenizer.encode("cabac") == [2, 0, 1, 0, 2]
        assert tokenizer.decode([2, 0, 1, 0, 2]) == "cabac"

    def test_unicode_and_empty_text(self) -> None:
        text = "caf\u00e9\U0001f642"
        tokenizer = CharTokenizer().fit(text)

        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert tokenizer.decode(tokenizer.encode("")) == ""

    def test_unknown_character(self) -> None:
        tokenizer = CharTokenizer().fit("abc")

        with pytest.raises(ValueError, match="characters not seen"):
            tokenizer.encode("d")

    def test_requires_fit(self) -> None:
        with pytest.raises(ValueError, match="fit must be called"):
            CharTokenizer().encode("a")


class TestBPETokenizer:
    def test_learns_greedy_pair_sequence_and_round_trips(self) -> None:
        tokenizer = BPETokenizer(max_merges=2).fit("abab")

        assert tokenizer.merges == [("a", "b"), ("ab", "ab")]
        assert tokenizer.vocab_size == 4
        assert tokenizer.decode(tokenizer.encode("abab")) == "abab"
        assert len(tokenizer.encode("abab")) == 1

    def test_ties_are_broken_lexicographically(self) -> None:
        tokenizer = BPETokenizer(max_merges=1).fit("abac")

        assert tokenizer.merges == [("a", "b")]

    def test_merge_limit_and_zero_merges(self) -> None:
        tokenizer = BPETokenizer(max_merges=0).fit("abab")

        assert tokenizer.merges == []
        assert len(tokenizer.encode("abab")) == 4

    def test_whitespace_and_newlines_are_regular_characters(self) -> None:
        text = "a a\nb"
        tokenizer = BPETokenizer(max_merges=10).fit(text)

        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_unknown_character_and_invalid_merge_limit(self) -> None:
        tokenizer = BPETokenizer(max_merges=1).fit("abab")
        with pytest.raises(ValueError, match="characters not seen"):
            tokenizer.encode("c")
        with pytest.raises(ValueError, match="non-negative"):
            BPETokenizer(max_merges=-1)
        with pytest.raises(TypeError, match="must be an integer"):
            BPETokenizer(max_merges=1.5)  # type: ignore[arg-type]

    def test_token_id_validation(self) -> None:
        tokenizer = BPETokenizer(max_merges=0).fit("a")

        with pytest.raises(ValueError, match="out of range"):
            tokenizer.decode([1])
        with pytest.raises(TypeError, match="integers"):
            tokenizer.decode([0.5])  # type: ignore[list-item]
