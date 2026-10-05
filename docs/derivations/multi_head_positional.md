# Multi-head attention and positional encodings

The derivation for the second M6 unit, walked through in chat and confirmed before
any code (`plan.md` §0.3). Like [attention.md](attention.md), everything here is built
on the autograd engine and defines **no backward of its own**, except one small new
primitive (`getitem`, §3). Notation follows [`docs/conventions.md`](../conventions.md).

| Symbol | Shape | Meaning |
|---|---|---|
| $d$ | | `d_model`, $d = h\,d_h$ |
| $h$, $d_h$ | | number of heads, per-head width |
| $X_q$ | $(B, T_q, d)$ | query-side input |
| $X_{kv}$ | $(B, T_k, d)$ | key/value-side input (equal to $X_q$ for self-attention) |
| $W^Q, W^K, W^V, W^O$ | $(d, d)$ | projections, each with a bias $b \in \mathbb{R}^d$ |
| $A_i$ | $(B, T_q, T_k)$ | attention weights of head $i$ |

## 1. Multi-head attention

One softmax over $T_k$ keys gives each query a single mixture. Heads let a query take
several mixtures at once, each from its own low-dimensional view of the inputs.

**Forward.**

1. Project: $Q = X_q W^Q + b^Q$, $K = X_{kv} W^K + b^K$, $V = X_{kv} W^V + b^V$,
   each of shape $(B, T, d)$.
2. Split into heads: head $i$ is columns $[i\,d_h,\,(i+1)\,d_h)$ of $Q, K, V$. In code,
   `reshape(B, T, h, d_h)` then `transpose(0, 2, 1, 3)` gives $(B, h, T, d_h)$, so
   the head axis is a batch axis.
3. Attend per head, with the scale $1/\sqrt{d_h}$ (not $1/\sqrt d$; the dot product
   in each head sums $d_h$ terms, see [attention.md](attention.md) §1):
   $$O_i = \operatorname{softmax}\!\Big(\frac{Q_iK_i^\top}{\sqrt{d_h}} + M\Big)V_i.$$
   The existing batched `scaled_dot_product_attention` does all heads in one call.
4. Merge: `transpose(0, 2, 1, 3)` then `reshape(B, T_q, d)` gives
   $\operatorname{concat}_i O_i$, and $\text{out} = \operatorname{concat}_i(O_i)\,W^O + b^O$.

**Why $d \times d$ projections.** Column slices of one $(d, d)$ matrix are exactly $h$
separate $(d, d_h)$ per-head projections $W^Q_i$, so the "full projection then split"
form is the per-head form of Vaswani et al. (2017), with one matmul instead of $h$.
`test_matches_per_head_loop` checks this against a literal Python loop over column
slices.

**Cost.** Each head costs $O(T_qT_kd_h)$, so all heads together cost
$O(T_qT_kd)$, the same as one head of full width. Parameters: $4d^2 + 4d$.

**Backward.** None to write. `reshape` and `transpose` already have VJPs (the inverse
reshape and the inverse permutation, [autograd.md](autograd.md) §4), as do the batched
`matmul` and `softmax` inside attention. The merge/split are exact inverses, so the
engine routes each head's gradient back to its own column slice of $W^Q, W^K, W^V$.

**Masks.** `scaled_dot_product_attention` expects a mask broadcastable to
$(B, h, T_q, T_k)$. A $(T_q, T_k)$ causal mask already is. A $(B, \cdot, T_k)$ padding
mask (and its `&` with a causal mask, $(B, T_q, T_k)$) is 3-D, so a head axis is
inserted, $(B, 1, \cdot, T_k)$; without it the batch axis would line up with the head
axis. A 4-D mask is passed through untouched, so a different mask per head is allowed.

**Interface.** `forward(q, k=None, v=None, mask=None) -> (output, weights)`, with
`k` defaulting to `q` and `v` to `k`: self-attention with one argument, cross-attention
with two or three. `weights` is per head, $(B, h, T_q, T_k)$. Initialization is Xavier
uniform weights and zero biases, as in `torch.nn.MultiheadAttention`.

## 2. Positional encodings

Attention is permutation-equivariant: permuting the rows of $X$ permutes the output
rows the same way, so nothing in the layer knows token order. A positional encoding
$P \in \mathbb{R}^{T \times d}$ is added to the embeddings, $X \mapsto X + P_{[:T]}$,
which breaks the symmetry.

**Sinusoidal** (Vaswani et al., 2017) is a fixed table:
$$P_{p,2i} = \sin(p\,\omega_i),\qquad P_{p,2i+1} = \cos(p\,\omega_i),\qquad
\omega_i = 10000^{-2i/d},\quad i = 0,\dots,\tfrac d2 - 1.$$
The wavelengths $2\pi/\omega_i$ grow geometrically from $2\pi$ to about $10000\cdot2\pi$,
so low columns resolve neighbouring positions and high columns distinguish far-apart
ones. The key property is that a shift is a rotation. By the angle-addition formulas,
for each $(\sin, \cos)$ pair,
$$\begin{pmatrix}P_{p+k,2i}\\P_{p+k,2i+1}\end{pmatrix}=
\begin{pmatrix}\cos k\omega_i & \sin k\omega_i\\ -\sin k\omega_i & \cos k\omega_i\end{pmatrix}
\begin{pmatrix}P_{p,2i}\\P_{p,2i+1}\end{pmatrix}$$
so $P_{p+k}$ is a *fixed linear function* of $P_p$ that depends only on the offset $k$,
which is what lets a linear layer express "attend $k$ steps back"
(`test_shift_is_a_rotation_per_frequency_pair`). It has no parameters and no gradient
path, so $\bar X = \bar O$. $d$ must be even (sines and cosines come in pairs).

**Learned** is a trainable table $P \in \mathbb{R}^{L_{\max}\times d}$ with the same
forward $X + P_{[:T]}$, initialized $\mathcal{N}(0, 0.02^2)$ (the GPT-2 choice). It
cannot extrapolate past $L_{\max}$ (a longer input raises), which sinusoidal can.
Its backward is the sum of the upstream gradient over the batch, placed in the first
$T$ rows:
$$\bar P_{[:T]} = \sum_b \bar O_b,\qquad \bar P_{[T:]} = 0.$$

## 3. The one new primitive: `getitem`

`P[:T]` needs indexing, which `ops.py` lacked. For $y = a[\text{idx}]$ the VJP scatters
the upstream gradient into zeros of $a$'s shape, $\bar a = \text{zeros}$,
$\bar a[\text{idx}] = g$. This is correct for **basic** indexing (ints and slices),
where no element is selected twice. Array indexing can repeat an element and would
need accumulation (`np.add.at`), so it is out of scope until something needs it
(an `Embedding` lookup in M7). It is exposed as `Tensor.__getitem__`.

## 4. Tests

- Multi-head: output equals a per-head Python loop; with one head it equals
  projected `scaled_dot_product_attention`; causal and padding masks hold in every head;
  a 4-D per-head mask passes through; shapes for cross-attention; every parameter
  receives a gradient; finite-difference checks on the inputs and all four weight
  matrices.
- `getitem`: matches NumPy and gradient-checks for ints, slices and `...`; the gradient
  is zero outside the slice.
- Sinusoidal: closed-form values, bounded in $[-1, 1]$, distinct rows, the rotation
  identity above.
- Learned: adds the first $T$ rows; gradient equals the batch sum in those rows and
  zero elsewhere, and matches finite differences.
- `-m reference`: `MultiHeadAttention` output, per-head weights, input gradient, $W^O$
  gradient and the stacked $W^Q, W^K, W^V$ gradients match
  `torch.nn.MultiheadAttention` (`batch_first=True`, `average_attn_weights=False`) for
  no mask, causal and padding. Torch stores `in_proj_weight` as the stacked transposed
  weights and its boolean mask means "blocked", the opposite of ours.
