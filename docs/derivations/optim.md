# Optimizers: SGD, Momentum, Nesterov, RMSprop, Adam

The finalized derivation walkthrough for `scratchgrad.optim`, written per
`plan.md` §0.3 *before* the implementation. Notation follows
[`docs/conventions.md`](../conventions.md).

Unlike every algorithm so far, this is not one model with one objective —
it is a **family of generic first-order update rules** for minimizing an
arbitrary differentiable $L(\theta)$, given only $g_t = \nabla_\theta
L(\theta_t)$ at each step. The gradient itself comes from elsewhere (a
classical estimator's own loss gradient today; a hand-derived backward pass
from M3's `nn.Module` onward; `autograd` from M5 onward) — `optim/` only
ever consumes $g_t$, never computes it.

## 1. Interface

```python
class SGD:  # and Momentum, Nesterov, RMSprop, Adam — same shape
    def __init__(self, lr: float = 0.01, ...) -> None: ...  # hyperparameters only
    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        ...  # mutates params[i] in place; state lazily allocated per-shape
             # on the first call, keyed by list position
```

`params`/`grads` are **lists** of arrays, not one flat vector: a future
`Linear` layer has two parameters of different shapes (`W` `(d_in,d_out)`,
`b` `(d_out,)`), and flattening/concatenating them just to satisfy a
single-vector optimizer is exactly the kind of premature machinery
`plan.md` §8 mistake 2 warns against. `step` mutates each `params[i]`
in place (`params[i] -= ...`, never rebinding it to a new array) —
the same shape as PyTorch's `optimizer.step()`, so whichever object holds
those arrays (a classical estimator's `theta`, or later a layer's `W`/`b`)
sees the update with no "read the result back" step.

**No shared `BaseOptimizer` class.** Per `plan.md` §8 mistake 2, an
abstract base is added only once the shared shape is *observed* across
real implementations, not guessed up front. All five are written out below
first; §7 revisits whether anything is actually worth factoring out, and
concludes no — the five `step` bodies differ enough (SGD: none; Momentum/
Nesterov: one buffer; RMSprop: one buffer; Adam: two buffers + a step
counter) that a shared base would either under- or over-fit them.

## 2. SGD (plain)

$$\boxed{\;\theta_{t+1} = \theta_t - \eta\, g_t\;}$$

The direct discretization of gradient descent on $L$. "Stochastic" refers
to how the *caller* computes $g_t$ (a full-batch loss gradient, as every
M1/M2 `solver="gd"` estimator does today, or a minibatch estimate of it,
once M3's `nn.Module` trainer exists) — nothing about `SGD.step` itself
changes between those two cases, so there is no separate "GD" class.

## 3. Momentum (classical / "heavy ball" — Polyak, 1964)

$$v_{t+1} = \mu\, v_t - \eta\, g_t, \qquad \theta_{t+1} = \theta_t + v_{t+1}$$

$v$ (one buffer per parameter, same shape, $v_0 = 0$) is an
exponentially-decaying running sum of past gradients. Unrolling the
recursion, $v_{t+1} = -\eta\sum_{k=0}^{t}\mu^{t-k} g_k$: a weighted sum of
all past gradients, most recent weighted highest. In a direction where
consecutive gradients agree in sign, the terms reinforce and the effective
step grows past a single $-\eta g_t$; in a direction where they oscillate
in sign, the terms partially cancel and the effective step shrinks. This
is the standard "ball rolling down a ravine, gaining speed along the
valley floor and damping across it" picture (Sutskever et al., 2013, §1).

$\mu \in [0, 1)$, default $0.9$. **$\mu = 0$ must reduce exactly to plain
SGD** — $v_{t+1} = -\eta g_t$, so $\theta_{t+1} = \theta_t - \eta g_t$,
identical to §2. This is a direct regression test (§8).

## 4. Nesterov accelerated gradient — the one genuine subtlety

Textbook NAG (Nesterov, 1983) evaluates the gradient at a **lookahead
point** $\theta_t + \mu v_t$, not at $\theta_t$ itself:

$$v_{t+1} = \mu v_t - \eta\, \nabla L(\theta_t + \mu v_t), \qquad
  \theta_{t+1} = \theta_t + v_{t+1}$$

That does not fit §1's interface: `grads` as passed to `step` is always
$\nabla L$ evaluated at the *current* `params`, and there is no mechanism
here — no autograd until M5 — to ask the caller to re-run forward+backward
at a shifted point in the middle of a single `step` call.

**Fix — Sutskever, Martens, Dahl & Hinton (2013), §2, eq. 7.** Substitute
the change of variable $\psi_t = \theta_t + \mu v_t$ (the lookahead point)
as the variable actually tracked from the outside. Since the caller always
computes gradients at whatever `params` currently holds, and under this
convention `params` *is* $\psi_t$, the gradient the caller passes in,
$g_t = \nabla L(\psi_t)$, is already exactly the quantity textbook NAG
needs — no shifted-point evaluation required anywhere.

**Derivation of the $\psi$-only recursion.** Start from
$\theta_{t+1} = \theta_t + v_{t+1}$ and $\psi_t = \theta_t + \mu v_t
\Rightarrow \theta_t = \psi_t - \mu v_t$:

$$\psi_{t+1} = \theta_{t+1} + \mu v_{t+1}
  = (\theta_t + v_{t+1}) + \mu v_{t+1}
  = \theta_t + (1+\mu) v_{t+1}$$

substitute $\theta_t = \psi_t - \mu v_t$:

$$\boxed{\;v_{t+1} = \mu v_t - \eta\, g_t, \qquad
  \psi_{t+1} = \psi_t - \mu v_t + (1+\mu)\, v_{t+1}\;}$$

Same state as Momentum (one buffer $v$), plus keeping the *previous* $v_t$
around for one extra term each step — not a new kind of state, just one
extra scalar array read before $v$ is overwritten.

**Correctness check for this substitution (not just an assertion — a
test, §8).** On a toy quadratic $L(\theta) = \tfrac12\theta^\top A\theta$
with a known closed-form gradient $\nabla L(\theta) = A\theta$, run the
*literal* lookahead formula directly (evaluate $\nabla L$ at the true
shifted point $\theta_t + \mu v_t$ each step, using the actual analytic
gradient function — something only possible in a test, where the gradient
function is known in closed form, not through the `step(params, grads)`
interface) side by side with the reformulated recursion above, and assert
the two trajectories match over many steps. This is what justifies
trusting the substitution rather than just citing the paper.

## 5. RMSprop (Hinton, *Coursera: Neural Networks for Machine Learning*,
lecture 6e, 2012 — unpublished; the standard citation for this method)

$$s_{t+1} = \beta s_t + (1-\beta) g_t^2, \qquad
  \theta_{t+1} = \theta_t - \eta\, \frac{g_t}{\sqrt{s_{t+1}} + \epsilon}$$

($g_t^2$ and the division are elementwise.) $s$ is a running average of
*squared* gradients — an elementwise estimate of each coordinate's recent
gradient scale. Dividing by $\sqrt{s_{t+1}}$ shrinks the step in
directions with large/noisy gradients and relatively grows it in
directions with small ones, which is what makes RMSprop robust to
badly-scaled or non-stationary objectives where a single global $\eta$
(plain SGD/Momentum) would need per-direction tuning. Default
$\beta = 0.9$, $\epsilon = 10^{-8}$ (Hinton's slides; matches PyTorch's
default $\epsilon$). $\epsilon$ is added *after* the square root — dividing
by a plain $\sqrt{s_{t+1}}$ would be undefined the first time a
coordinate's gradient is exactly $0$.

**A property worth noting, caught while writing the convergence test
(§8): with a *constant* $\eta$, RMSprop's step size does not shrink as
$\theta$ approaches the optimum.** Dividing $g_t$ by $\sqrt{s_{t+1}}$ — its
own recent magnitude — makes each step roughly constant size $\approx
\eta$ *regardless of how large or small $g_t$ itself is*, unlike a plain
gradient step, which shrinks automatically near a bowl's minimum as
$g_t \to 0$. Two consequences: progress toward the optimum is steady
rather than accelerating (more like sign-gradient descent than classic
gradient descent), and once close, $\theta$ settles into a residual
oscillation ("noise ball") around the minimum whose radius scales with
$\eta$ (measured empirically: radius $\approx 0.7\eta$ on a simple 2D
bowl) rather than shrinking further with more steps. Not a bug — this is
why RMSprop/Adam are typically paired with a learning-rate schedule in
practice; the convergence test uses a small $\eta$ and enough steps to
cover the initial distance, not just a longer run at a fixed $\eta$.

Plain/uncentered, no built-in momentum term — some libraries (PyTorch,
TensorFlow) offer `centered=True` (subtract a running mean of $g$ before
squaring) or a `momentum` blend; both are extensions on top of Hinton's
original minimal form, out of scope here (`plan.md` §8 mistake 2 — add
only once a concrete need shows up).

## 6. Adam (Kingma & Ba, 2015, *Adam: A Method for Stochastic
Optimization*)

$$m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t, \qquad
  s_{t+1} = \beta_2 s_t + (1-\beta_2) g_t^2$$

$m$ is Momentum's running mean of gradients (§3, but as an average, not an
unnormalized sum); $s$ is RMSprop's running mean of squared gradients
(§5). $m_0 = s_0 = 0$ biases both estimates toward zero, especially in the
first few steps when $\beta_1, \beta_2$ are close to $1$ — **bias
correction** removes exactly that bias (Kingma & Ba, §3, via the geometric
series identity $\mathbb{E}[s_t] = (1-\beta_2^t)\,\mathbb{E}[g^2] + \dots$
under a stationarity assumption on $g$):

$$\hat m = \frac{m_{t+1}}{1-\beta_1^{t+1}}, \qquad
  \hat s = \frac{s_{t+1}}{1-\beta_2^{t+1}}$$

$$\boxed{\;\theta_{t+1} = \theta_t - \eta\, \frac{\hat m}{\sqrt{\hat s}+\epsilon}\;}$$

$t$ starts at $0$ for the first `step` call (so $1-\beta_1^{t+1}$ starts
away from $0$ at $t=0$, matching the paper's algorithm box). Default
$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ — the paper's own
defaults, also PyTorch's. **No `AdamW` / decoupled weight decay** —
`plan.md`'s M5 `autograd/optim.py` line explicitly scopes `AdamW` to
arrive once `Tensor` parameters exist; adding it here would be exactly
the "decide a rule before a concrete need shows up" mistake (§8 mistake
16's general shape, applied to weight decay instead of dtype).

## 7. Why no shared base class (revisited after all five)

| | state | update shape |
| --- | --- | --- |
| SGD | none | $\theta -= \eta g$ |
| Momentum | $v$ | $v = \mu v - \eta g;\ \theta += v$ |
| Nesterov | $v$ (+ previous $v$) | $v = \mu v - \eta g;\ \theta -= \mu v_{\text{prev}} - (1+\mu)v$ |
| RMSprop | $s$ | $s = \beta s + (1-\beta)g^2;\ \theta -= \eta g/(\sqrt{s}+\epsilon)$ |
| Adam | $m, s, t$ | bias-corrected combination of the Momentum and RMSprop shapes |

The only literal duplication is "lazily allocate one zero-filled buffer
per parameter shape on the first `step` call", three lines, repeated in
Momentum/Nesterov/RMSprop/Adam (SGD needs none). Per `plan.md` §8 mistake
2, three duplicated lines across four files is not the kind of shared
shape worth an abstract base class over — a `BaseOptimizer` would need to
either impose a state schema that doesn't fit SGD (no state) and Nesterov
(two buffers, not one) equally well, or become a thin, pointless wrapper.
Revisit if a sixth optimizer's state genuinely repeats one of these
exactly.

## 8. Testing plan

| Tier | Check |
| --- | --- |
| Analytic | One hand-computed `step` per optimizer on fixed toy `params`/`grads` (exact arithmetic, e.g. SGD: `θ - lr*g`; Adam: manual bias-corrected first step). |
| Reduction | `Momentum(momentum=0)` produces byte-identical updates to `SGD` at the same `lr`, over several steps. |
| Substitution correctness | Nesterov's reformulated recursion (§4) matches the literal lookahead formula (gradient evaluated at the true shifted point via a known closed-form `grad_fn`) on a toy quadratic, over many steps. |
| Convergence/behavioral | Each optimizer minimizes a well-conditioned quadratic bowl to near-zero loss; on a deliberately **ill-conditioned** elongated quadratic (condition number `>> 1`), Momentum/RMSprop/Adam reach a given loss threshold in fewer steps than plain SGD at a matched `lr` — the standard demonstration of what each addition buys, and `examples/optim.py`'s punchline. |
| Contract | Invalid hyperparameters raise `ValueError` (`lr<=0`; `momentum` outside `[0,1)`; `beta`/`beta1`/`beta2` outside `(0,1)`; `eps<=0`); state buffers match each `params[i]`'s shape after the first `step`; hyperparameters are never mutated by `step`. |
| Determinism | Two identical sequences of `step` calls from the same initial state produce byte-identical `params`. |
| Reference (`-m reference`, new: PyTorch) | Feed identical `(params, grads)` sequences to this project's optimizer and to `torch.optim.SGD` / `torch.optim.SGD(momentum=..., nesterov=True)` / `torch.optim.RMSprop` / `torch.optim.Adam`; assert the resulting parameter arrays match to `atol` at every step — this is a genuine exact-parity case (identical published formulas), not a tolerance-only outcome comparison like `LinearSVM`/`RandomForest`. |

Plus `examples/optim.py`: the ill-conditioned-quadratic race, printing
each optimizer's loss after a fixed step budget.

## References

- Polyak, B.T. (1964), "Some methods of speeding up the convergence of
  iteration methods", *USSR Computational Mathematics and Mathematical
  Physics* 4(5) — classical momentum.
- Nesterov, Y. (1983), "A method for solving the convex programming
  problem with convergence rate $O(1/k^2)$" — the original accelerated
  gradient method.
- Sutskever, I., Martens, J., Dahl, G. & Hinton, G. (2013), "On the
  importance of initialization and momentum in deep learning", *ICML* —
  the gradient-only-at-current-point reformulation of NAG used in §4.
- Hinton, G. (2012), *Coursera: Neural Networks for Machine Learning*,
  lecture 6e — RMSprop.
- Kingma, D.P. & Ba, J. (2015), "Adam: A Method for Stochastic
  Optimization", *ICLR*.
