# Tiny GPT on tiny Shakespeare

The M7 capstone derivation was walked through in chat and confirmed before code,
as required by [`plan.md`](../../plan.md). This example uses the existing
decoder-only Pre-LN transformer, autograd, and character tokenizer. The only new
gradient rule is for repeated array indices in an embedding lookup. Notation
follows [`docs/conventions.md`](../conventions.md).

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $B,T,V,d$ | | Batch size, context length, vocabulary size, model width |
| $X,Y$ | $(B,T)$ | Input character IDs and next-character targets |
| $E,P$ | $(V,d),(T,d)$ | Token and learned position tables |
| $Z$ | $(B,T,V)$ | Next-character logits |

## Model and objective

A training window of $T+1$ characters becomes $X_{b,t}=c_{b,t}$ and
$Y_{b,t}=c_{b,t+1}$. At position $t$, the model may read only positions
$0,\ldots,t$:

$$H^{(0)}_{b,t}=E_{X_{b,t}}+P_t,\qquad
H^{(L)}=\operatorname{Decoder}_{\text{causal}}(H^{(0)}),\qquad
Z_{b,t}=H^{(L)}_{b,t}W+b.$$

The decoder is `Decoder(cross_attention=False)`, so there is no encoder
memory or unused cross-attention parameter. Its blocks use causal multi-head
attention and Pre-LN residual branches; the stack applies a final LayerNorm.
The output head has its own matrix $W$ (weights are not tied to $E$).

For $p_{b,t,v}=\operatorname{softmax}(Z_{b,t})_v$, minimize mean next-character
cross-entropy over all $BT$ predictions:

$$L=-\frac{1}{BT}\sum_{b,t}\log p_{b,t,Y_{b,t}}
=-\frac{1}{BT}\sum_{b,t,v}\mathbf{1}[Y_{b,t}=v]\log p_{b,t,v}.$$

Differentiating the log-softmax gives

$$\frac{\partial L}{\partial Z_{b,t,v}}
=\frac{p_{b,t,v}-\mathbf{1}[Y_{b,t}=v]}{BT}.$$

The example flattens logits to $(BT,V)$ and passes one-hot targets to the
existing stable `cross_entropy`, whose mean is therefore over all positions.
The existing autograd engine propagates this gradient through the head,
decoder, and positional addition. For the token table, each occurrence of ID
$k$ contributes to the same row:

$$\frac{\partial L}{\partial E_k}
=\sum_{b,t:X_{b,t}=k}\frac{\partial L}{\partial H^{(0)}_{b,t}}.$$

Likewise, the gradient of each position row $P_t$ sums over the batch. NumPy
advanced indexing can repeat an ID, so `Tensor.__getitem__` uses `np.add.at`
in its vector-Jacobian product. Plain indexed assignment would keep only the
last contribution. There is no hand-written backward in `Embedding` or
`TinyGPT`.

## Training and generation

```text
fit the character vocabulary and encode the corpus
split the encoded corpus into train (first 90%) and validation (last 10%)
initialize token table, position table, causal decoder, and output head
repeat for each training step:
    sample B contiguous windows of T+1 characters from the training split
    predict logits for the first T characters
    compute mean cross-entropy against the following T characters
    clear parameter gradients; backpropagate; take one Adam step
periodically estimate validation loss on fresh windows
for generation, repeatedly predict from the latest T generated characters,
sample from the final-position softmax, and append the sampled character
```

The runnable script is [`examples/tiny_gpt.py`](../../examples/tiny_gpt.py).
It downloads tiny Shakespeare once through the existing loader and caches the
corpus. It seeds model initialization, batch selection, validation sampling,
and text sampling separately. The train/validation split is contiguous, so
their target windows do not overlap. Fitting the character vocabulary on the
whole corpus fixes the ID mapping; it does not use validation targets for
parameter updates.

## Measured run

Run with `python -u examples/tiny_gpt.py` on 2026-10-06. This deliberately
small CPU model uses 32-character contexts, batch size 8, one decoder block,
model width 32, four heads, feed-forward width 128, no dropout, Adam at
learning rate 0.003, and 2,000 updates. All trainable arrays use `float64`.
Validation values are means of four sampled batches at each report step.

| Step | Training loss | Validation loss |
| ---: | ---: | ---: |
| 1 | 4.6040 | 4.3118 |
| 500 | 2.4705 | 2.4160 |
| 1,000 | 2.2101 | 2.2396 |
| 1,500 | 2.1376 | 2.2601 |
| 2,000 | 2.0820 | 2.1090 |

A sampled continuation began:

> ROMEO: I thas'I beeasow sil your my thath sepoather, thou bret of caster thy purr ie,

The sample has character patterns and speaker-like formatting but remains
mostly nonsensical. That matches the limited capacity and training budget;
the falling held-out loss is the stronger evidence of learning. The script
saves the trained parameters and vocabulary to the ignored local path
`.cache/tiny-gpt/model.npz`. The checkpoint is reproducible by running the
script; it is not committed to the repository.

To reuse the checkpoint without retraining, construct the same model and copy
its parameter arrays in `parameters()` order:

```python
import numpy as np

from scratchgrad.transformer import CharTokenizer, TinyGPT

with np.load(".cache/tiny-gpt/model.npz") as saved:
    tokenizer = CharTokenizer().fit("".join(saved["vocabulary"]))
    model = TinyGPT(tokenizer.vocab_size, 32, 32, 4, 1, 128, random_state=0)
    for i, parameter in enumerate(model.parameters()):
        parameter.data[...] = saved[f"param_{i}"]
model.eval()
```

## Verification

Tests check that repeated embedding IDs sum their gradients, including a
central finite-difference check; that all model parameters receive gradients
and an optimizer step changes the embedding; and that changing future tokens
does not change earlier logits. Earlier transformer tests already check the
decoder, attention, LayerNorm, and their gradients independently.
