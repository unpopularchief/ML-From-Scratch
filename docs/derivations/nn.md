# `nn`: `Module`, `Linear`, activations, losses, init schemes

The finalized derivation walkthrough for `scratchgrad.nn`, written per
`plan.md` §0.3 *before* the implementation, and reconfirmed in chat
(handoff.md §6) since a layer is neither a `fit`/`predict` estimator
(`base.Estimator`) nor a parameter-only optimizer (`optim/`) — it needs its
own `forward`/`backward` shape. Notation follows
[`docs/conventions.md`](../conventions.md): `x` a layer's input, `y` its
output, `W`/`b` weights/bias, `n`/`d` sample/feature counts.

Like `optim/` (`docs/derivations/optim.md` §1), this is not one model with
one objective — it is the shared machinery every hand-derived MLP layer,
activation, and loss in M3/M4 is built from. `optim/` consumes a gradient;
`nn/` is what computes one, one layer at a time, by the chain rule.

## 1. `Module` interface

```python
class Module:
    def forward(self, x: FloatArray) -> FloatArray: ...
    def backward(self, grad_output: FloatArray) -> FloatArray: ...
    def parameters(self) -> list[FloatArray]: ...  # default: []
    def grads(self) -> list[FloatArray]: ...  # default: []
```

`forward(x)` computes `y = f(x)` and caches whatever `backward` needs.
`backward(grad_output)` receives the upstream gradient
$\bar y = \partial L/\partial y$ and returns $\bar x = \partial L/\partial
x$ via the chain rule — for a parameterized layer it also computes and
stashes $\partial L/\partial(\text{each parameter})$, retrievable through
`grads()`. `parameters()`/`grads()` return **same-order, same-shape**
lists — this is deliberate: it is exactly the `list[FloatArray]` shape
`optim.step(params, grads)` already consumes (PR #20), so
`opt.step(layer.parameters(), layer.grads())` needs no adapter. A
parameterless `Module` (every activation here) inherits the `[]` default
from both.

Plain base class, not an ABC, **not** subclassing `base.Estimator` — same
reasoning `optim/`'s derivation already established: `conventions.md`
defines "estimator" as "has `fit`", and a layer has neither `fit` nor
`predict`.

**No `training`/`eval` mode flag yet.** `Dropout`/`BatchNorm` (the next
ROADMAP line, same milestone) will need one, but nothing in this unit
does — added when that PR actually consumes it (`plan.md` §8 mistake 2:
no shared base machinery ahead of an observed need, the same call already
made for `optim/`'s missing `BaseOptimizer`).

**No `Sequential` container yet either.** Chaining layers by hand (call
each `forward` in order, each `backward` in reverse, collect
`parameters()`/`grads()` across the chain) is enough to gradient-check
every piece individually and to build `examples/nn.py`'s small hand-wired
MLP. A composing container is naturally motivated once the trainer /
MNIST example (both separate, later ROADMAP lines) need to stop hand-
wiring a specific network shape.

## 2. `Linear`

$$Y = XW + b, \qquad X: (n, d_{in}),\ W: (d_{in}, d_{out}),\ b: (d_{out},),\ Y: (n, d_{out})$$

`W` is stored `(d_in, d_out)`, not PyTorch's `(d_out, d_in)` — this
matches the project's `X: (n_samples, n_features)` convention used
everywhere else, so `Y = XW + b` reads directly off the shapes without a
transpose; the PyTorch reference test transposes `nn.Linear.weight` when
comparing.

Backward, from the upstream $\bar Y = \partial L/\partial Y$, $(n,
d_{out})$: each entry $Y_{ij} = \sum_k X_{ik}W_{kj} + b_j$, so by the
chain rule, summing over every $Y$ entry a given parameter/input entry
feeds into:

$$\frac{\partial L}{\partial W_{kj}} = \sum_i \bar Y_{ij} X_{ik}
  \;\Rightarrow\; \bar W = X^\top \bar Y \quad (d_{in}, d_{out})$$

$$\frac{\partial L}{\partial b_j} = \sum_i \bar Y_{ij}
  \;\Rightarrow\; \bar b = \mathbf{1}^\top \bar Y \quad (d_{out},)
  \text{ — column sum over samples}$$

$$\frac{\partial L}{\partial X_{ik}} = \sum_j \bar Y_{ij} W_{kj}
  \;\Rightarrow\; \bar X = \bar Y W^\top \quad (n, d_{in})$$

$\bar X$ is what gets passed to the previous layer's `backward`.

## 3. Activations

All four are elementwise, so `backward` is always `grad_output *
local_derivative` (a diagonal Jacobian) — except `Softmax`, whose outputs
all depend on every input in a row, giving a genuinely dense per-row
Jacobian.

- **ReLU** $f(x) = \max(0, x)$: $f'(x) = \mathbb{1}[x>0]$. Subgradient $0$
  at the kink $x=0$ (matches PyTorch's convention; a continuous-valued
  input hits $x=0$ with probability 0 in every gradcheck anyway).
- **Sigmoid** $f(x) = \sigma(x)$ (reuses `utils.math.sigmoid`):
  $f'(x) = \sigma(x)(1-\sigma(x))$, computed from the cached *output*
  $y=\sigma(x)$, not by recomputing $\sigma$.
- **Tanh** $f(x) = \tanh(x)$: $f'(x) = 1-\tanh^2(x) = 1-y^2$, same
  cached-output trick.
- **Softmax** (last axis) $f(x)_i = e^{x_i}/\sum_j e^{x_j}$ (reuses
  `utils.math.softmax`). Per-row Jacobian
  $J = \mathrm{diag}(y) - yy^\top$ (standard softmax-Jacobian identity,
  $y=f(x)$), so $\bar x = J\bar y = y\odot\bar y - y\,(y\cdot\bar y)
  = y\odot(\bar y - (y\cdot\bar y))$ — computed batched via one
  row-wise dot product (`(y * grad_output).sum(-1, keepdims=True)`), no
  explicit $(d,d)$ Jacobian matrix ever materialized.

  Standalone `Softmax` is rarely used directly (almost always fused
  inside `CrossEntropyLoss` for stability, §4) but implements this full
  backward anyway rather than staying forward-only — not much extra code,
  and it keeps `Softmax` independently gradient-checkable like every
  other activation here (`plan.md` §2: "gradient checks for **every**
  layer, activation, and loss").

## 4. Losses

Each loss takes **raw scores** directly — logits for `BCEWithLogitsLoss`/
`CrossEntropyLoss`, i.e. fused internally with the link function — rather
than an already-activated probability from a preceding `Sigmoid`/
`Softmax`. This is the same stability move `LogisticRegression` already
makes (`softplus(z) - yz` instead of `-log(sigmoid(z))`), and matches
PyTorch's actual default loss classes (`BCEWithLogitsLoss`,
`CrossEntropyLoss` — not `BCELoss`/`NLLLoss`, which take probabilities
and are the numerically fragile path PyTorch itself recommends against).

A loss is **not** a `Module` subclass: it is always the last op ("root")
of a backward pass, `forward` takes two arguments (`y_pred`, `y_true`)
rather than `Module`'s one, and `backward` takes **no** upstream gradient
— there is nothing further downstream to hand one to. Mirrors the same
"don't force a shape that doesn't fit" call `optim/`'s derivation made
for skipping a `BaseOptimizer`.

- **MSE** $L = \mathrm{mean}(( \hat y - y)^2)$, mean over **every** entry
  of `y_pred` (samples *and* output dimensions together, not samples
  only) — matches PyTorch's `nn.MSELoss()` default, this unit's `-m
  reference` parity target. $\bar y_{pred} = 2(\hat y - y)/|\hat y|$
  ($|\hat y|$ = total element count, i.e. $n \cdot d_{out}$).
- **`BCEWithLogitsLoss`** $L = \mathrm{mean}(\mathrm{softplus}(z) - yz)$,
  $\mathrm{softplus}(z) = \log(1+e^z)$ via `np.logaddexp(0, z)` for
  stability. $\partial L/\partial z = (\sigma(z) - y)/n$ — the softplus
  derivative is $\sigma(z)$, so this is a direct generalization of
  `LogisticRegression`'s own loss gradient.
- **`CrossEntropyLoss`** (multiclass; `y_true` one-hot, $(n,K)$; logits
  $Z$, $(n,K)$) $L = \mathrm{mean}_i(-\log\mathrm{softmax}(Z_i)[y_i])$,
  computed as $-Z_{i,y_i} + \mathrm{logsumexp}(Z_i)$ via
  `utils.math.logsumexp` (never $\log(\mathrm{softmax}(Z))$ directly,
  which underflows to $-\infty$ then `nan` whenever a class probability
  rounds to exactly 0). $\partial L/\partial Z = (\mathrm{softmax}(Z) -
  y_{true})/n$ — the standard softmax-cross-entropy gradient identity,
  re-derived here rather than assumed: writing $L_i = -Z_{i,y_i} +
  \log\sum_k e^{Z_{ik}}$, $\partial L_i/\partial Z_{ij} =
  -\mathbb{1}[j=y_i] + \mathrm{softmax}(Z_i)_j$, which is exactly
  $(\mathrm{softmax}(Z_i) - \text{onehot}(y_i))_j$.

## 5. Init schemes

All take an explicit `rng: numpy.random.Generator` (`zeros` doesn't need
one, but the other two follow `conventions.md`'s "randomness always flows
through an explicit `random_state`" rule via `check_random_state`, applied
at `Linear.__init__` — `nn/init.py`'s functions themselves take an
already-resolved `Generator`, the same layering `cluster`/`ensemble`
already use between an estimator's `random_state` param and its
module-level init/split helpers).

- **`zeros(shape)`**: all-zero. Used for `b` always — a bias has no
  symmetry problem (every unit's bias is independent). **Never** used for
  a weight matrix feeding more than one output unit: every unit would
  compute an identical function of the input and receive an identical
  gradient forever (the standard symmetry-breaking argument), so `Linear`
  only calls this for `b`, never `W`, in normal use — `weight_init="zeros"`
  is exposed for completeness/testing (e.g. confirming *why* it's a bad
  default is worth being able to construct), not recommended.
- **`xavier_uniform(fan_in, fan_out, rng)`** (Glorot & Bengio, 2010):
  $W\sim U(-a,a)$, $a=\sqrt{6/(fan_{in}+fan_{out})}$. Derived from
  requiring $\mathrm{Var}(y_j) = \mathrm{Var}(x_i)$ forward *and*
  $\mathrm{Var}(\bar x_i)=\mathrm{Var}(\bar y_j)$ backward through a
  linear/tanh-like (odd, near-identity-at-0) activation — with $d_{in}$
  independent inputs of variance $\sigma^2$, $\mathrm{Var}(y_j) =
  d_{in}\sigma^2\mathrm{Var}(x)$ (each weight independent of $x$, zero
  mean), so $\sigma^2=1/d_{in}$ preserves forward variance and
  $\sigma^2=1/d_{out}$ preserves backward variance; splitting the
  difference, $\sigma^2 = 2/(d_{in}+d_{out})$, gives a uniform
  distribution's variance $a^2/3$ set equal to that.
- **`he_normal(fan_in, fan_out, rng)`** (He, Zhang, Ren & Sun, 2015):
  $W\sim\mathcal N(0, 2/d_{in})$. Same forward-variance argument as
  Xavier's $\sigma^2=1/d_{in}$ term, but ReLU zeroes out (in expectation)
  half its input, halving the variance that survives to the next layer —
  doubling $\sigma^2$ to $2/d_{in}$ exactly cancels that factor. `Linear`
  defaults to this (`weight_init="he"`) since ReLU is the default
  activation pairing assumed in the M3 MLP example.

## 6. Testing plan

| Tier | Check |
| --- | --- |
| Gradient check | Central-difference gradcheck (`tests/helpers/gradcheck.py`) for every module-level `_*_backward` against its `_*_forward`, individually — `Linear`'s `dX`/`dW`/`db` each checked via a fixed random upstream direction `R` (`f(X) = sum(_linear_forward(X,W,b) * R)`, etc.), every activation, and every loss. `plan.md` §2's highest-value test. |
| Analytic | Hand-computed forward/backward on tiny fixed arrays (e.g. `ReLU` on a fixed sign pattern; `MSELoss` on a 2-point fixed error). |
| Reduction identities | `Momentum`-style base-case check, `nn` analogue: `he_normal`'s and `xavier_uniform`'s *empirical* mean/variance over many draws matches the closed-form target (`mean≈0`, `var≈2/fan_in` / `var≈2/(fan_in+fan_out)`) within a statistical tolerance. `Softmax` backward reduces to the same value as the direct `CrossEntropyLoss` gradient when composed (`softmax` then a one-hot cross-entropy-of-probabilities check). |
| Contract | `parameters()`/`grads()` same length, same-shape, same order; non-positive `in_features`/`out_features` raise; hyperparameters not mutated by `forward`/`backward`; unknown `weight_init` raises. |
| Determinism | Same `random_state` → byte-identical `Linear.W`; two identical forward/backward sequences from the same start produce byte-identical outputs and grads. |
| Integration | `examples/nn.py`: a hand-wired `Linear → ReLU → Linear → BCEWithLogitsLoss` MLP, trained with `optim.Adam` on `make_moons`, reaches a real accuracy improvement over a fixed budget — the `nn`/`optim` combination working end-to-end, this unit's punchline (the M3-so-far analogue of `optim.md`'s ill-conditioned-bowl race). |
| Reference (`-m reference`, PyTorch) | Feed identical `x`/parameters through `Linear`/each activation/each loss and `torch.nn.Linear`/`torch.nn.functional.*`/`torch.nn.*Loss`; forward outputs and `backward()`-derived gradients compared at `atol` — **exact parity expected** (identical formulas, same tier as `optim/`), not tolerance-only. |

## References

- Glorot, X. & Bengio, Y. (2010), "Understanding the difficulty of
  training deep feedforward neural networks", *AISTATS* — Xavier
  initialization.
- He, K., Zhang, X., Ren, S. & Sun, J. (2015), "Delving Deep into
  Rectifiers: Surpassing Human-Level Performance on ImageNet
  Classification", *ICCV* — He/Kaiming initialization.
