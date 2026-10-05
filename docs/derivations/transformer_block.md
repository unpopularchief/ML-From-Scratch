# Pre-LN transformer block, encoder and decoder

The derivation for the first M7 unit, walked through in chat and confirmed before any
code (`plan.md` §0.3). Like [attention.md](attention.md) and
[multi_head_positional.md](multi_head_positional.md), everything here is built on the
autograd engine and defines **no backward of its own**: LayerNorm and GELU are forward
compositions of existing ops, and the closed-form LayerNorm gradient below is used only
as a test oracle. Notation follows [`docs/conventions.md`](../conventions.md).

| Symbol | Shape | Meaning |
|---|---|---|
| $d$, $d_{ff}$ | | `d_model`, feed-forward width (usually $4d$) |
| $X$ | $(B, T, d)$ | residual stream |
| $\gamma, \beta$ | $(d)$ | LayerNorm scale and shift |
| $M$ | $(B, T_m, d)$ | encoder output ("memory") read by cross-attention |

## 1. LayerNorm

Normalize each position's feature vector on its own, so statistics never mix across
the batch or the sequence (unlike `BatchNorm1d`):

$$\mu = \tfrac1d\sum_j x_j,\qquad \sigma^2 = \tfrac1d\sum_j (x_j-\mu)^2,\qquad
\hat x = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y = \gamma\odot\hat x+\beta.$$

The variance is the biased one, as in `torch.nn.functional.layer_norm`. In code it is
`centered * (var + eps) ** -0.5 * gamma + beta`: `mean`, `sub`, `mul`, `pow` and
broadcasting, all with VJPs already, so `Tensor.backward` differentiates it.

**Gradient check by hand.** With $\delta=\partial L/\partial y$ and
$\hat\delta=\delta\odot\gamma$, the chain rule through $\mu$ and $\sigma^2$ (every $x_j$
feeds both) gives

$$\frac{\partial L}{\partial x} = \frac{1}{\sigma_\epsilon}\Big(\hat\delta - \operatorname{mean}(\hat\delta)
- \hat x\,\operatorname{mean}(\hat\delta\odot\hat x)\Big),\quad
\frac{\partial L}{\partial\gamma}=\sum \delta\odot\hat x,\quad
\frac{\partial L}{\partial\beta}=\sum\delta,$$

with $\sigma_\epsilon=\sqrt{\sigma^2+\epsilon}$ and the sums over every axis but the last.
This is the `BatchNorm1d` train-mode formula ([dropout_batchnorm.md](dropout_batchnorm.md)
§3) with the mean taken over features instead of the batch. The test suite asserts the
engine's gradient equals it to $10^{-10}$.

## 2. GELU

$$\operatorname{GELU}(x)=x\,\Phi(x)\approx\tfrac12x\Big(1+\tanh\big(\sqrt{2/\pi}\,(x+0.044715\,x^3)\big)\Big).$$

The exact form needs $\operatorname{erf}$, which would be a new primitive with its own
VJP (and no `scipy.special` at runtime). The tanh form composes from `tanh`, `mul`,
`pow` and matches `gelu(approximate="tanh")` in PyTorch exactly; it is also what GPT-2
used.

## 3. Feed-forward and the Pre-LN block

$$\operatorname{FFN}(x)=\operatorname{GELU}(xW_1+b_1)W_2+b_2,\quad W_1\in\mathbb R^{d\times d_{ff}},
\ W_2\in\mathbb R^{d_{ff}\times d},$$

applied independently at each position (so it never mixes positions; the tests check
this). The Pre-LN residual block is

$$X' = X + \operatorname{Drop}\big(\operatorname{MHA}(\operatorname{LN}(X))\big),\qquad
X'' = X' + \operatorname{Drop}\big(\operatorname{FFN}(\operatorname{LN}(X'))\big).$$

**Why Pre-LN.** Post-LN, $\operatorname{LN}(X+f(X))$, puts a normalization on the
residual path, so every layer's Jacobian multiplies into the gradient on the way down.
In Pre-LN the path from output to input is a pure sum,

$$\frac{\partial X''}{\partial X} = I + \frac{\partial}{\partial X}\big[\text{branch terms}\big],$$

so the identity term carries the gradient straight through all layers. Training is
stable without a learning-rate warmup. The price: the stream's scale grows with depth,
which is why the stack ends with one more LayerNorm (§5). Dropout sits on each branch,
before it is added back; it is the identity in `eval()`.

With the branch output projections ($W^O$ and $W_2$) set to zero a block is exactly the
identity, which the tests use as a structural check.

## 4. Decoder block

$$\begin{aligned}
X' &= X + \operatorname{Drop}\big(\operatorname{MHA}_{\text{causal}}(\operatorname{LN}_1(X))\big)\\
X'' &= X' + \operatorname{Drop}\big(\operatorname{MHA}(\operatorname{LN}_c(X')\,;\,M,M)\big)
\quad\text{(only if cross-attending)}\\
X''' &= X'' + \operatorname{Drop}\big(\operatorname{FFN}(\operatorname{LN}_2(X''))\big).
\end{aligned}$$

Self-attention defaults to the causal mask (position $i$ sees keys $j\le i$); a caller
can pass a combined causal-and-padding mask instead. In cross-attention queries come
from the decoder stream and keys and values from $M$, so $T_q\ne T_k$ is fine and a
`padding_mask` over $M$ blocks padded source positions. **Decision (confirmed):** the
cross-attention sublayer exists only when the block is built with
`cross_attention=True`. With `False` the block has no cross-attention parameters at all
(a parameter that never gets a gradient would trip an optimizer), and it is the
decoder-only block the tiny GPT of M7 uses.

## 5. Stacks

An `Encoder` is $N$ Pre-LN blocks followed by a final LayerNorm, run bidirectionally
under a padding mask. A `Decoder` is $N$ decoder blocks and a final LayerNorm. The final
LayerNorm is required because the Pre-LN residual stream is unnormalized: its variance
grows with every block, so whatever reads the output (a language-model head,
cross-attention in a decoder) needs normalized features.

**Decision (confirmed):** the stacks take already-embedded $(B, T, d)$ tensors. Token
`Embedding` (which needs array indexing in autograd) and positional-encoding wiring
arrive with the tokenizer and GPT units.

## 6. Verification

- LayerNorm and GELU: formula, the closed-form gradient above, finite differences, and
  PyTorch parity for outputs and all gradients.
- Blocks: the structural equations above reproduced by hand from the sublayers,
  identity when branches are zeroed, mask behaviour (padded or future positions cannot
  influence the rest), every parameter receives a gradient, finite-difference input
  gradients.
- PyTorch: `EncoderBlock` against `nn.TransformerEncoderLayer(norm_first=True)` and
  `DecoderBlock` against `nn.TransformerDecoderLayer(norm_first=True)` (tanh GELU,
  `dropout=0`), weights copied across; outputs and input/memory gradients match to
  $10^{-10}$.
