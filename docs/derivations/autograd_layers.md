# `autograd`: functional ops, layers, optimizers on `Tensor`

The derivation for the second M5 unit, walked through in chat before any code
(`plan.md` §0.3). Unit 1 ([autograd.md](autograd.md)) built the engine; this
unit builds the three things that sit on top of it: `autograd/functional.py`,
`autograd/layers.py` and `autograd/optim.py`. The point is the same as the
`nn/`-versus-`autograd/` duplication `plan.md` §1 describes: `nn/` writes each
layer's backward by hand, here **no layer or loss has a backward at all**. It is
a forward composition of primitive ops, and `Tensor.backward()` supplies the
rest. Notation follows [`docs/conventions.md`](../conventions.md).

## 1. Layers need no new backward math

`Linear` is `y = x @ W + b` with `W`, `b` leaf Tensors (`requires_grad=True`).
The `matmul` VJP gives $\bar W = x^\top \bar y$ and $\bar x = \bar y\, W^\top$;
the broadcast `+ b` is un-summed by `_unbroadcast` into $\bar b = \sum_i \bar y_i$.
These are exactly `nn.Linear.backward`'s three lines
([nn.md](nn.md) §2), now produced by the engine. `W` is `(d_in, d_out)` and is
drawn from the same `nn.init` functions with the same seed, so an autograd
`Linear` and an `nn.Linear` start from *identical* weights, which is what makes
the parity tests (§5) exact.

`Dropout` multiplies by a constant keep mask $m = \text{bernoulli}(1-p)/(1-p)$,
so the mask is a no-grad Tensor operand and $\bar x = \bar y \odot m$
([dropout_batchnorm.md](dropout_batchnorm.md) §2) falls out of `mul`'s VJP.
Eval mode is the identity and draws nothing.

Parameters are leaves whose `.grad` **accumulates** (unit 1's choice), so every
training step is `zero_grad()` → forward → `backward()` → `step()`.

## 2. Stable functional ops from unstable primitives

The primitives are `exp`/`log`, which overflow or underflow on their own.
Each stable form below is the same algebra `nn/losses.py` uses, and the
stabilizing shift is a **constant** (a detached/no-grad array), which is
allowed because the shifted expression equals the unshifted one for every
shift value, so the true gradient does not depend on it.

- **`log_softmax`:** $\log\operatorname{softmax}(z)_j = z_j - \operatorname{lse}(z)$,
  $\operatorname{lse}(z) = c + \log\sum_k e^{z_k-c}$, $c=\max_k z_k$ as a
  constant. Differentiating through the ops gives
  $\partial \operatorname{lse}/\partial z_j = \operatorname{softmax}(z)_j$ automatically,
  so the Jacobian $\operatorname{diag}(y)-yy^\top$ is never written down.
- **`softmax`** $=\exp(\texttt{log\_softmax})$.
- **`cross_entropy(Z, Y)`** with one-hot `Y`: $L=-\frac1n\sum_{i,j}Y_{ij}\log\operatorname{softmax}(Z_i)_j$.
  Same one-hot convention as `nn.CrossEntropyLoss`, so no gather/index primitive
  is needed: the "select the true class" is `(Y * log_softmax).sum()`.
- **`bce_with_logits(z, y)`** $=\operatorname{mean}(\operatorname{softplus}(z)-yz)$ with
  $\operatorname{softplus}(z)=m+\log(e^{-m}+e^{z-m})$, $m=\max(z,0)$ constant.
  Its derivative is $e^{z-m}/(e^{-m}+e^{z-m})=\sigma(z)$, matching `nn`.
- **`mse_loss`** $=\operatorname{mean}((\hat y-y)^2)$ over every element, matching
  `nn.MSELoss`.

`gelu` and `layer_norm` (named in `plan.md`) are **deferred**: nothing in M5
needs them, and M6 (attention) is where they are first used.

## 3. Optimizers over `Tensor` parameters

An optimizer holds the parameter Tensors and reads `p.grad` on `step()`,
updating `p.data` **in place** with no graph recorded. It works on raw
`ndarray`s, so no global `no_grad()` is needed (unit 1's decision stands).
A parameter whose `.grad` is `None` (not reached by the last `backward()`) is
skipped, as in PyTorch.

The update rules are the ones already derived in [optim.md](optim.md); only the
interface changes (`step()` instead of `step(params, grads)`):

- **`SGD`:** $\theta \leftarrow \theta - \eta g$.
- **`Adam`:** $m\leftarrow\beta_1 m+(1-\beta_1)g$, $s\leftarrow\beta_2 s+(1-\beta_2)g^2$,
  $\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat s}+\epsilon)$ with
  $\hat m=m/(1-\beta_1^t)$, $\hat s=s/(1-\beta_2^t)$.
- **`AdamW`** (Loshchilov & Hutter, 2019): **decoupled** weight decay.
  Adding $\lambda\theta$ to the gradient (L2 regularization) makes the decay pass
  through $\hat m/\sqrt{\hat s}$, so heavily-updated parameters are decayed
  *less*. AdamW instead shrinks the weights directly,
  $\theta\leftarrow\theta(1-\eta\lambda)$, then applies the plain Adam step with
  the *unmodified* gradient. Same order as `torch.optim.AdamW`.

`Momentum`, `Nesterov` and `RMSprop` are not repeated here; they add no new
idea over M3's `optim/` and nothing downstream needs them.

## 4. Scope and limits

- `Sequential` takes layers (`Module`s) and plain callables (e.g. `ops.relu`),
  so activations need no wrapper class. `parameters()` and `train()`/`eval()`
  recurse into the `Module` children only.
- Minimal `Module`: `parameters()`, `zero_grad()`, `train()`/`eval()`,
  `__call__`. The `nn.Module` `forward`/`backward`/`grads` contract does not
  apply (there is no `backward`), so this is a separate class that reuses only
  `nn.init`.
- Cross-entropy takes **one-hot** targets, like `nn`.

## 5. Verification plan

1. Gradient-check every functional op against `tests/helpers/gradcheck.py`.
2. **Parity with the hand-derived `nn/`:** the same seeded `Linear`, same
   `X`/`Y`, loss gradients and parameter gradients equal `nn`'s to
   `atol=1e-10`; likewise `Dropout` with the same mask and each optimizer
   against `optim/` over several steps.
3. Edge cases: parameters with `None` grad are skipped, eval-mode `Dropout` is
   the identity, extreme logits stay finite, invalid hyperparameters raise.

The M5 proof test (a whole MLP matching the hand-derived gradients) is the
next unit; the parity tests above are its building blocks.
