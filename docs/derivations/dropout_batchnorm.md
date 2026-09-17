# `Module.training`/`eval`, `Dropout`, `BatchNorm1d`

The finalized derivation walkthrough for the third M3 `nn` unit, written per
`plan.md` §0.3 *before* the implementation and confirmed in chat (handoff.md
§6 had already flagged that `Module` would need a `training`/`eval` flag
once this unit actually consumed one — deferred rather than guessed at when
`nn.md` was written). Notation follows
[`docs/conventions.md`](../conventions.md); this unit also introduces
`p` (dropout probability), `gamma`/`beta` (BatchNorm's learnable scale/
shift), `mu`/`var` (batch mean/variance), `eps`, `momentum`.

Three genuinely open interface questions were confirmed via
`AskUserQuestion` before any code, all answered "recommended": (1) a
stateful `self.training` flag toggled by `train()`/`eval()` methods on
`Module` (mirrors `torch.nn.Module`), not an explicit `training=` argument
threaded through every `forward()` call; (2) `BatchNorm1d.running_var`
tracks the Bessel-corrected (unbiased) variance, exactly matching
`torch.nn.BatchNorm1d`, even though the batch itself is normalized with the
biased variance; (3) `Dropout` takes a `random_state` constructor argument
seeding an internal `Generator` that keeps advancing across calls, per
`conventions.md`'s "randomness always flows through an explicit
`random_state`" rule.

## 1. `Module.training` / `train()` / `eval()`

```python
class Module:
    training: bool = True

    def train(self) -> Module:
        self.training = True
        return self

    def eval(self) -> Module:
        self.training = False
        return self
```

`training` is a **class attribute**, not set in an `__init__` — every
existing subclass (`Linear`, every activation) gets it for free without
needing to call `super().__init__()`, the same way the empty-list
`parameters()`/`grads()` defaults already work. Calling `train()`/`eval()`
sets an *instance* attribute that shadows the class default.

Only `Dropout` and `BatchNorm1d` read `self.training`; every other `Module`
in this package ignores it completely — no behavior change for `Linear` or
any activation. There is still no `Sequential` container (deferred, same
reasoning as `nn.md` §1), so a hand-wired network calls `.eval()`/`.train()`
on each stateful layer instance directly, exactly how `examples/nn.py`
already hand-chains `forward`/`backward`.

## 2. `Dropout(p=0.5, random_state=None)`

**Inverted dropout** (the modern convention, matches `torch.nn.Dropout`):
scale *during training* by `1/(1-p)` so evaluation needs no compensating
rescale.

**Training forward:** draw a per-element Bernoulli keep mask with
probability `1-p`, scaled by `1/(1-p)`:

$$m_i \sim \mathrm{Bernoulli}(1-p) \big/ (1-p), \qquad y_i = x_i \cdot m_i$$

$$\mathbb E[y_i] = x_i \cdot \mathbb E[m_i] = x_i \cdot \frac{(1-p)}{(1-p)} = x_i$$

— this is the whole point of the `1/(1-p)` scale: the *expected* output
during training already equals `x`, so `eval()` can be a plain identity
with no rescale needed at inference time (the "vanilla" dropout convention
instead scales by `1-p` at eval time — equivalent in expectation, but
means every inference call pays an extra multiply; inverted dropout moves
that cost to the (rarer, training-only) mask draw).

**Training backward:** `y = x * m` is elementwise-linear in `x` for a
*fixed* `m` (the mask doesn't depend on `x`), so directly
$\partial L/\partial x_i = \partial L/\partial y_i \cdot m_i$ — the same
mask reused from `forward`.

**Eval forward/backward:** identity both ways, `y=x`,
$\partial L/\partial x = \partial L/\partial y$. No RNG draw at all in eval
mode — the generator's state is untouched, so switching to eval and back to
train resumes the same draw sequence eval never consumed.

$p$ must be in $[0,1)$ — $p=1$ divides by zero in $1/(1-p)$; raises
`ValueError`. Parameterless (`[]` default `parameters()`/`grads()`, same as
every activation).

## 3. `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`

Over `(n, d)` input, matching `Linear`'s output shape — feature-wise
normalization over the batch dimension. Spatial `BatchNorm2d` deferred to
M4 alongside `Conv2d` (nothing to normalize over spatially yet). Ioffe &
Szegedy (2015).

### Forward (training)

$$\mu = \frac1n\sum_i x_i, \quad
  \mathrm{var} = \frac1n\sum_i (x_i-\mu)^2 \ \text{(biased, } n\text{, not } n-1),
  \quad \mathrm{std}=\sqrt{\mathrm{var}+\epsilon}$$

$$\hat x_i = \frac{x_i-\mu}{\mathrm{std}}, \qquad y_i = \gamma \hat x_i + \beta$$

(all per-feature; $n$ = batch size, applied independently to each of the
$d$ columns.) Then update the running buffers used at eval time:

$$\text{running\_mean} \leftarrow (1-\text{momentum})\cdot\text{running\_mean} + \text{momentum}\cdot\mu$$

$$\text{running\_var} \leftarrow (1-\text{momentum})\cdot\text{running\_var} + \text{momentum}\cdot\left(\mathrm{var}\cdot\frac{n}{n-1}\right)$$

**The Bessel correction is the one real subtlety here, confirmed as a
scoping question rather than assumed:** the batch is normalized with the
*biased* variance (dividing by $n$), but `running_var` accumulates the
*unbiased* estimate (dividing by $n-1$) — `torch.nn.BatchNorm1d` does
exactly this (its own docs note "the standard-deviation is calculated via
the biased estimator... however, the value... used for normalization is
calculated via the unbiased estimator" for the running stats specifically).
Matched here for exact reference-test parity. This also means training
mode requires $n>1$ (division by $n-1$) — `BatchNorm1d` raises `ValueError`
for a training-mode batch of size 1, matching PyTorch's own error for that
case.

### Forward (eval)

$$\hat x = \frac{x-\text{running\_mean}}{\sqrt{\text{running\_var}+\epsilon}},
  \qquad y = \gamma\hat x+\beta$$

No buffer updates. `running_mean`/`running_var` are **not** learnable —
not returned by `parameters()`/`grads()`, only `gamma`/`beta` are.

### Backward (training) — the full chain-rule derivation

$\gamma$/$\beta$ are direct: $y_i=\gamma\hat x_i+\beta$ is linear in each,
so

$$\frac{\partial L}{\partial\gamma}=\sum_i \bar y_i \hat x_i, \qquad
  \frac{\partial L}{\partial\beta}=\sum_i \bar y_i$$

with $\bar y = \partial L/\partial y$. Let $\overline{\hat x}_i = \bar
y_i\gamma$ (direct, same linearity). The subtlety: $\hat x_k$ depends on
**every** $x_i$ in the batch, not just $x_k$, because $\mu$ and
$\mathrm{var}$ are batch statistics. So
$\partial L/\partial x_i = \sum_k \overline{\hat x}_k\, \partial \hat
x_k/\partial x_i$ needs the *full* Jacobian of $\hat x$ w.r.t. $x_i$, not
just the diagonal term.

Working out $\partial\hat x_k/\partial x_i$ directly (writing $\delta_{ki}$
for the Kronecker delta, dropping the per-feature index since every feature
is independent):

$$\hat x_k=\frac{x_k-\mu}{\mathrm{std}}, \qquad
  \frac{\partial\mu}{\partial x_i}=\frac1n, \qquad
  \frac{\partial\mathrm{var}}{\partial x_i}
    = \frac2n\sum_j(x_j-\mu)\Big(\delta_{ji}-\frac1n\Big)
    = \frac2n(x_i-\mu)$$

(the $\sum_j(x_j-\mu)\cdot(-1/n)$ term vanishes since $\sum_j(x_j-\mu)=0$),
so $\partial\mathrm{std}/\partial x_i = (x_i-\mu)/(n\cdot\mathrm{std})$, and

$$\frac{\partial\hat x_k}{\partial x_i}
  = \frac{\delta_{ki}-1/n}{\mathrm{std}} - \frac{(x_k-\mu)}{\mathrm{std}^2}\cdot\frac{x_i-\mu}{n\cdot\mathrm{std}}
  = \frac1{\mathrm{std}}\Big(\delta_{ki}-\frac1n-\frac{\hat x_k\hat x_i}{n}\Big)$$

Summing over $k$ with $\overline{\hat x}_k$:

$$\frac{\partial L}{\partial x_i}
  = \frac1{\mathrm{std}}\Big(\overline{\hat x}_i - \frac1n\sum_k\overline{\hat x}_k
    - \hat x_i\cdot\frac1n\sum_k\overline{\hat x}_k\hat x_k\Big)
  = \frac{\overline{\hat x}_i - \mathrm{mean}(\overline{\hat x}) - \hat x_i\cdot\mathrm{mean}(\overline{\hat x}\odot\hat x)}{\mathrm{std}}$$

This is the standard batchnorm backward closed form (re-derived here from
the chain rule, not just cited — the same discipline `nn.md` used for the
Softmax Jacobian and the cross-entropy gradient identity), implemented in
`_batchnorm_backward_train` exactly as written (with
$\overline{\hat x}=\bar y\odot\gamma$ substituted in).

### Backward (eval) — a genuinely different, simpler formula

In eval mode $\mu=\text{running\_mean}$ and
$\mathrm{std}=\sqrt{\text{running\_var}+\epsilon}$ are **fixed constants**,
not functions of the current batch — so $\hat x_i$ depends only on $x_i$
itself, and the cross-sample coupling above disappears entirely:

$$\frac{\partial L}{\partial x_i} = \bar y_i\cdot\frac{\gamma}{\mathrm{std}}$$

`BatchNorm1d.backward` dispatches on which mode the *matching* `forward`
call ran in (cached at `forward` time, not re-read from `self.training`,
in case `train()`/`eval()` is toggled between the two calls) — this is a
real behavioral difference worth its own gradient check, not just a detail:
the eval-mode backward is algebraically a different, simpler function, not
an approximation of the training one.

## 4. Testing plan

| Tier | Check |
| --- | --- |
| Gradient check | `Dropout` train-mode backward against a fixed mask (`f(x)=sum(_dropout_forward(x,mask)*R)`); `BatchNorm1d` train-mode `dX`/`dgamma`/`dbeta` *and* eval-mode `dX`/`dgamma`/`dbeta` separately, each via a fixed upstream direction `R`, the same trick `nn.md` §6 used for `Linear`/activations. |
| Analytic | Hand-computed `BatchNorm1d` forward on a tiny fixed batch (verify output has exactly zero mean / unit variance per feature before the `gamma`/`beta` affine, given `gamma=1,beta=0`). |
| Reduction identity | `Dropout`'s empirical `E[mask]≈1` and `E[y]≈x` over many training-mode draws (the inverted-dropout property itself), the `nn`-unit analogue of `he_normal`/`xavier_uniform`'s moment checks. |
| Contract | `training` defaults `True`; `train()`/`eval()` toggle and return `self`; `Dropout`'s `p` outside `[0,1)` raises; `BatchNorm1d`'s non-positive `num_features` raises, training-mode batch size 1 raises; `parameters()`/`grads()` same length/shape/order for `BatchNorm1d` (`Dropout` stays parameterless). |
| Determinism | Same `Dropout` `random_state` -> byte-identical mask sequence across calls; `eval()` mode never advances the generator. |
| Integration | `examples/nn.py` extended with `BatchNorm1d`/`Dropout` in the hidden layer, explicit `.train()` before the training loop and `.eval()` before computing held-out-style accuracy — demonstrating the flag actually changes the forward pass, not just bookkeeping. |
| Reference (`-m reference`, PyTorch) | `BatchNorm1d`: identical `x`/`gamma`/`beta` through both sides in training mode *and* eval mode (after copying `running_mean`/`running_var`), comparing forward output, backward gradients, and the post-step `running_mean`/`running_var` -- exact parity expected (same tier as `optim`/`nn.md`'s other components). `Dropout`: RNG streams differ between NumPy and PyTorch, so this is eval-mode-only exact parity (`Dropout` in eval is a pure identity on both sides) plus a statistical check in training mode (empirical keep-rate within tolerance of `1-p`). |

## References

- Ioffe, S. & Szegedy, C. (2015), "Batch Normalization: Accelerating Deep
  Network Training by Reducing Internal Covariate Shift", *ICML*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov,
  R. (2014), "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting", *JMLR* 15.
