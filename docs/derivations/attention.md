# Scaled dot-product attention, causal + padding masks

The derivation for the first M6 unit, walked through in chat and confirmed before
any code (`plan.md` §0.3). Attention is built entirely on the autograd engine
([autograd.md](autograd.md)): like the layers in
[autograd_layers.md](autograd_layers.md) it defines **no backward of its own**.
Notation follows [`docs/conventions.md`](../conventions.md).

| Symbol | Shape | Meaning |
|---|---|---|
| $Q$ | $(B, T_q, d_k)$ | queries |
| $K$ | $(B, T_k, d_k)$ | keys |
| $V$ | $(B, T_k, d_v)$ | values |
| $M$ | broadcastable to $(B, T_q, T_k)$ | additive mask (0 = allowed) |
| $A$ | $(B, T_q, T_k)$ | attention weights, rows sum to 1 |
| $O$ | $(B, T_q, d_v)$ | output |

## 1. Forward

$$S = \frac{QK^\top}{\sqrt{d_k}},\qquad A = \operatorname{softmax}(S + M)\ \text{(over the key axis)},\qquad O = AV.$$

**Why $\sqrt{d_k}$.** For independent zero-mean, unit-variance entries,
$q\cdot k=\sum_{i=1}^{d_k} q_ik_i$ has variance $d_k$. Unscaled, the logits' spread
grows with $d_k$, softmax saturates toward one-hot and its gradient
$\operatorname{diag}(a)-aa^\top$ vanishes. Dividing by $\sqrt{d_k}$ restores unit
variance (`test_scale_keeps_logit_variance_near_one`).

## 2. Backward, and the one new primitive

Every step is an existing differentiable op except one gap: `ops.matmul` was
2-D only, and attention is batched. It is generalised to `np.matmul` semantics
(leading axes batch and broadcast, at least 2 dims per operand). For
$C = XY$ with upstream $\bar C$ the VJPs are unchanged,

$$\bar X = \bar C\,Y^\top,\qquad \bar Y = X^\top \bar C,$$

with the transposes on the last two axes only, followed by `_unbroadcast` to sum
out any batch axis that was broadcast (e.g. a shared $(d, k)$ weight against a
$(B, T, d)$ input gets its gradient summed over $B$ and $T$-batches).

Composing the ops and writing $\bar O$ for the upstream gradient, the engine
produces what a hand derivation gives:

$$\bar V = A^\top \bar O,\qquad \bar A = \bar O V^\top,\qquad
\bar S = A\odot\big(\bar A - \textstyle\sum_j \bar A_{\cdot j}A_{\cdot j}\big),$$
$$\bar Q = \frac{\bar S K}{\sqrt{d_k}},\qquad \bar K = \frac{\bar S^\top Q}{\sqrt{d_k}}.$$

The $\bar S$ line is the softmax Jacobian-vector product
($\operatorname{diag}(a)-aa^\top$ applied to $\bar A$); `softmax` in
`autograd/functional.py` already gets it for free from `exp`/`log`/`sum`.
`test_autograd_matches_hand_derived_gradients` checks all five to $10^{-10}$.

## 3. Masks

Masks are boolean arrays, `True` = the query may attend to the key. Inside the
function they become the additive constant $M_{ij}=0$ (allowed) or $-10^9$
(blocked), added to $S$ with no gradient path.

- **Causal**, `causal_mask(T_q, T_k)`: $M_{ij}$ blocked for $j>i$ (lower
  triangular), so position $i$ depends only on positions $\le i$. Shape
  $(T_q, T_k)$, broadcast over the batch.
- **Padding**, `padding_mask(lengths, T_k)`: blocks key $j$ of sequence $b$ when
  $j\ge \text{lengths}_b$. Shape $(B, 1, T_k)$, broadcast over queries.
- **Combined:** `causal_mask(...) & padding_mask(...)`.

A blocked score is $\approx -10^9$, so $e^{s-\max}$ underflows to exactly $0$:
the weight is $0$ and, because $\bar S=A\odot(\cdots)$, so are the gradients
into that key and value (`test_masked_positions_get_no_key_or_value_gradient`).
Padding equals truncating the sequence (`test_padding_equals_truncating_the_sequence`).

**Fully blocked rows.** With $-\infty$ a row with no allowed key would be
$0/0=$ NaN. A finite $-10^9$ keeps it finite, but the shift cancels in softmax,
so such a row's weights follow the raw scores $S$ (not uniform): finite, but
meaningless. Callers should not rely on those rows (e.g. a zero-length sequence).

## 4. Tests

- Forward equals the plain NumPy formula; rows sum to 1; equal keys give uniform
  weights.
- Causal weights are zero above the diagonal and the output ignores future values.
- Gradient check (finite differences) for $Q,K,V$, masked and unmasked.
- Autograd gradients equal the hand-derived formulas of §2.
- Batched `matmul` VJPs gradient-checked, including broadcast batch axes.
- `-m reference`: output and $\bar Q,\bar K,\bar V$ match
  `torch.nn.functional.scaled_dot_product_attention` for no mask, causal,
  padding and both.
