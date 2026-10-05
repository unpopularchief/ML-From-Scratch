# Character tokenizer and minimal BPE

The second M7 unit adds token IDs for text. Character tokenization is a useful
baseline: each Python Unicode code point is one token. BPE (byte-pair encoding)
then greedily replaces frequent adjacent token pairs with a new token. This
minimal version operates on code points rather than UTF-8 bytes and does not
split text with a regex pre-tokenizer. It can therefore learn merges containing
spaces and across word boundaries. Notation follows
[`docs/conventions.md`](../conventions.md).

## Character vocabulary

For training text $x$, let $C = \{x_i : x_i \text{ is a code point in } x\}$.
Sort $C$ by code point and assign IDs $0,\ldots,|C|-1$. Encoding maps each
code point through this table; decoding concatenates the corresponding code
points. Sorting makes the mapping independent of the order in which characters
first appeared. This implementation does not normalize combining characters.

## Greedy pair merges

Start with the training text as a list of character tokens. At iteration $k$,
count each adjacent pair $(a,b)$ over the current token sequence (including
pairs across spaces). Select

$$ (a_k,b_k)=\arg\max_{(a,b)}\;\mathrm{count}(a,b). $$

If several pairs have the same count, the lexicographically smallest pair of
token strings wins. Replace every non-overlapping occurrence of that pair from
left to right with the concatenated token $a_kb_k$. Stop when there are no
adjacent pairs or the requested merge limit is reached. The ordered pairs are
the learned merge rules. Their order is important: encoding starts from
characters and applies the same rules in order.

The algorithm is frequency-greedy rather than a differentiable optimization:
each step reduces the token count of the training sequence by the number of
replacements, but it does not claim to find a globally optimal vocabulary.

```text
tokens = characters(training_text)
for each allowed merge:
    counts = frequency of every adjacent pair in tokens
    if counts is empty: stop
    pair = highest-frequency pair, breaking ties lexicographically
    tokens = replace non-overlapping occurrences of pair with concatenate(pair)
    save pair
```

## Verification cases

- `"abab"` has pair `("a", "b")` twice, so the first merge is `"ab"`; the
  remaining pair `("ab", "ab")` is the second merge.
- `"abac"` gives `("a", "b")`, `("a", "c")`, and `("b", "a")` equal
  frequency; the stable tie-break selects `("a", "b")`.
- Character and BPE encodings decode back to their original text, including
  spaces, newlines, and Unicode code points. Unknown input characters and
  invalid token IDs are rejected explicitly.
