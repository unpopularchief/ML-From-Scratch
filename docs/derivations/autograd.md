# `autograd`: `Tensor`, topological `backward()`, ops + VJPs

The finalized derivation walkthrough for the first M5 unit, written per
`plan.md` §0.3 *before* the implementation and confirmed in chat. Where `nn/`
shows the chain rule written by hand layer by layer, `autograd/` automates it:
the same gradients, derived once per primitive op instead of once per layer.
Notation follows [`docs/conventions.md`](../conventions.md); `L` is a scalar
loss, `y` a node's output, and $\bar y = \partial L/\partial y$ its **adjoint**
(the upstream gradient).

## 1. `Tensor`

A `Tensor` holds `data` (float64 ndarray), `grad` (same shape, `None` until a
`backward()` reaches it), `requires_grad`, and two graph fields: `_parents`
(the input Tensors of the op that produced it) and `_vjp` (a closure,
below). A Tensor with no parents is a **leaf**.

Graph recording is skipped entirely when no input requires grad: the output
is a plain constant, so constants never grow a graph.

## 2. The VJP rule

For a node $y = f(x_1, \dots, x_k)$, reverse mode needs only

$$\bar x_i \mathrel{+}= \bar y \cdot \frac{\partial y}{\partial x_i}$$

as a **vector-Jacobian product**: the Jacobian is never materialized, each op
just supplies the closure `g -> (g_1, ..., g_k)`. The `+=` is the multivariate
chain rule: if a tensor feeds several consumers (`x*x`, or a diamond
`(x+1)*(x+2)`), its adjoint is the *sum* of the contributions from each use.

## 3. Topological `backward()`

An adjoint $\bar x$ is final only after every consumer of $x$ has pushed its
contribution. Visiting nodes in an order where each node follows all of its
consumers guarantees that: the **reverse** of a topological order.

1. DFS **post-order** from the root, visiting only grad-requiring parents,
   with a `visited` set of `id`s: every node appears after all its parents,
   exactly once (so a diamond is not processed twice).
2. Reset the adjoints of non-leaf nodes, seed `root.grad = 1`.
3. Walk the order **reversed**, calling each node's `_vjp` and accumulating
   into its parents.

The DFS is **iterative** (an explicit stack of `(node, children_done)`): a
recursive one would hit Python's recursion limit on a long chain such as an
unrolled RNN. A test builds a 5000-op chain.

The root must be scalar unless `grad=` is passed: `dL/dL = 1` only makes sense
for a scalar `L`.

**Accumulation semantics** (chosen, matching PyTorch): leaf `.grad` *accumulates*
across `backward()` calls; `zero_grad()` resets it. Intermediate nodes are
reset at the start of each call. Without that, with the graph kept alive
(also chosen: no `retain_graph` flag), a second `backward()` would propagate
the stale first-pass adjoint as well and over-count.

## 4. Op VJPs

With $g = \bar y$:

| Op | $y$ | VJP |
|---|---|---|
| add | $a + b$ | $\bar a = g,\ \bar b = g$ |
| sub | $a - b$ | $\bar a = g,\ \bar b = -g$ |
| mul | $a \odot b$ | $\bar a = g \odot b,\ \bar b = g \odot a$ |
| div | $a / b$ | $\bar a = g / b,\ \bar b = -g\, a / b^2$ |
| neg | $-a$ | $-g$ |
| pow | $a^p$ ($p$ constant) | $g \cdot p\, a^{p-1}$ |
| matmul | $AB$ | $\bar A = g B^\top,\ \bar B = A^\top g$ |
| exp | $e^a$ | $g \odot y$ |
| log | $\ln a$ | $g / a$ |
| tanh | $\tanh a$ | $g \odot (1 - y^2)$ |
| sigmoid | $\sigma(a)$ | $g \odot y(1 - y)$ |
| relu | $\max(a, 0)$ | $g \odot [a > 0]$ (subgradient 0 at 0) |
| sum | $\sum_{\text{axis}} a$ | $g$ broadcast back to $a$'s shape |
| mean | $\frac1N \sum a$ | same, divided by $N$ (number of elements reduced) |
| reshape | | $g$ reshaped back to $a$'s shape |
| transpose(perm) | | $g$ transposed by the inverse permutation (`argsort(perm)`) |

`matmul` is exactly `nn.Linear`'s backward ($\bar X = \bar Y W^\top$,
$\bar W = X^\top \bar Y$); `sum`'s broadcast-back is `Linear`'s
$\bar b = \bar Y$ summed over the batch, seen from the other side.
`matmul` is restricted to 2-D operands (batched matmul is deferred until
something needs it); `pow` takes a constant exponent only.

### Broadcasting

If `a: (n, d)` and `b: (d,)`, then `a + b` broadcasts `b` across rows, so the
adjoint of `b` must be the sum over those rows. `_unbroadcast(g, shape)` sums
away the extra leading axes, then sums (with `keepdims`) every axis where the
target size is 1 but `g`'s is not. It is applied to both operands of every
binary elementwise op.

## 5. API choices (confirmed)

- Operators (`+ - * / ** @`, unary `-`, reflected forms) and methods
  (`.exp() .log() .tanh() .sigmoid() .relu() .sum() .mean() .reshape() .T`)
  on `Tensor`; the VJPs live in `autograd/ops.py`. Python scalars and
  ndarrays on the left work (`__array_ufunc__ = None` makes `ndarray * Tensor`
  defer to the Tensor).
- `detach()` only; no global `no_grad()` until Tensor-based layers/optimizers
  need one (`plan.md` §8 mistake 2).
- Leaf grads accumulate, `zero_grad()` resets, the graph is kept after
  `backward()`.

## 6. Test plan

- Every op's VJP gradient-checked with `tests/helpers/gradcheck.py` (fixed
  random direction `R`, `f(x) = sum(op(x) * R)`), binary ops across
  broadcasting shape pairs, including a 0-d operand.
- Graph mechanics: fan-out, diamond, accumulation across calls, no
  intermediate double-counting, topological-order invariants, 5000-deep chain.
- `backward()` contract: scalar default seed, explicit seed, shape errors.
- A `tanh(XW + b)` / mean-square composite compared against the hand-derived
  `nn`-style gradients: the M5 proof test in miniature (the full MLP
  version belongs to the layers/optimizers unit).
