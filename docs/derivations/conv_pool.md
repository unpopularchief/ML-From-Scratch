# Convolution, max pooling, and flattening

The finalized derivation for M4's first unit, written and confirmed before
implementation as required by `plan.md` section 0.3. These layers keep the
manual `Module.forward`/`Module.backward` contract established in M3.

Images use **NCHW** layout throughout:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(N,C_{in},H,W)$ | input batch |
| $K$ | $(C_{out},C_{in},K_h,K_w)$ | convolution filters |
| $b$ | $(C_{out},)$ | optional bias |
| $Y$ | $(N,C_{out},H_{out},W_{out})$ | convolution output |

## 1. `Conv2d` forward

Deep-learning "convolution" is cross-correlation: the learned filter is not
spatially flipped. For symmetric padding $(P_h,P_w)$ and stride $(S_h,S_w)$,

$$
Y_{n,o,i,j}=b_o+\sum_{c=0}^{C_{in}-1}\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}K_{o,c,u,v}
X^{pad}_{n,c,iS_h+u,jS_w+v}.
$$

The spatial output size is

$$
H_{out}=\left\lfloor\frac{H+2P_h-K_h}{S_h}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W+2P_w-K_w}{S_w}\right\rfloor+1.
$$

Only numeric symmetric padding is in scope. Dilation, groups, and string
padding modes are later conveniences built on the same operation, not new
backpropagation lessons.

## 2. im2col

For every $(n,i,j)$ output position, flatten its
$(C_{in},K_h,K_w)$ receptive field into one row. Stacking those rows gives

$$
X_{col}:(NH_{out}W_{out},\ C_{in}K_hK_w).
$$

Flatten each filter into a row as well:

$$
K_{col}:(C_{out},\ C_{in}K_hK_w).
$$

The nested sum from section 1 is now an ordinary matrix multiplication:

$$
Y_{col}=X_{col}K_{col}^{\mathsf T}+b,
$$

where NumPy broadcasts $b$ over rows. Reshaping and transposing $Y_{col}$
returns NCHW output. The implementation builds patches with explicit spatial
loops and vectorizes across the batch/channels; this keeps the mapping from
the equation visible and avoids an opaque stride-trick view.

## 3. `Conv2d` backward

Let $G_{col}$ be the upstream gradient $\partial L/\partial Y$, rearranged
into the same row order as $Y_{col}$. The forward pass is now the same affine
map as `Linear`, so the three gradients follow directly:

$$
\frac{\partial L}{\partial K_{col}}=G_{col}^{\mathsf T}X_{col},
\qquad
\frac{\partial L}{\partial b}=\sum_r G_{col,r},
\qquad
\frac{\partial L}{\partial X_{col}}=G_{col}K_{col}.
$$

The first result reshapes to $(C_{out},C_{in},K_h,K_w)$. The final result is
not yet $\partial L/\partial X$: one input pixel can occur in several im2col
rows whenever receptive fields overlap. `col2im` therefore scatters every
patch gradient back to its original padded coordinates with **addition**.
After all contributions accumulate, it crops the padding.

Filter initialization uses convolutional fan counts:

$$
fan_{in}=C_{in}K_hK_w,
\qquad
fan_{out}=C_{out}K_hK_w.
$$

Thus He initialization has variance $2/fan_{in}$, while Xavier uniform uses
the limit $\sqrt{6/(fan_{in}+fan_{out})}$.

## 4. `MaxPool2d`

For each channel independently,

$$
Y_{n,c,i,j}=\max_{u,v}
X^{pad}_{n,c,iS_h+u,jS_w+v},
$$

with padded entries treated as $-\infty$, not zero (zero would incorrectly
win for an all-negative boundary window). Forward caches the flattened
row-major index $q$ selected by `argmax` in every window. Backward applies
the max subgradient

$$
\frac{\partial Y}{\partial x_q}=1,
\qquad
\frac{\partial Y}{\partial x_k}=0\quad(k\ne q),
$$

so each upstream value is scattered only to its cached winner. As in
`col2im`, overlapping windows add their contributions. At an exact tie the
maximum is non-differentiable and several subgradients are valid; this layer
deterministically chooses the first maximum, matching NumPy and PyTorch.

`MaxPool2d` has no learnable parameters. `stride=None` means
`stride=kernel_size`, giving non-overlapping windows by default. Dilation,
ceil-mode output sizing, and returned indices are outside this unit's scope.

## 5. `Flatten`

For input $X:(N,d_1,\ldots,d_k)$,

$$
Y=\operatorname{reshape}\left(X,\left(N,\prod_i d_i\right)\right).
$$

Reshape changes only the view/indexing, not any values, so its Jacobian is a
permutation of the identity. Backward reshapes the upstream gradient to the
cached input shape. `Flatten` preserves the batch dimension and has no
parameters.

## 6. Testing plan

| Tier | Check |
| --- | --- |
| Analytic | Tiny hand-computed convolution/pooling outputs; cross-correlation orientation; overlapping-gradient accumulation; flatten shape restoration. |
| Gradient check | Central finite differences for `Conv2d`'s $dX/dK/db$, `MaxPool2d`'s $dX$ on inputs with unique maxima, and `Flatten`'s reshape backward. |
| Contract | Invalid channels/spatial parameters, input shapes, bias-disabled parameter lists, deterministic initialization, and parameter/gradient ordering. |
| Reference | Identical NCHW inputs, filters, upstream gradients, stride, and padding compared with PyTorch `Conv2d`, `MaxPool2d`, and `flatten`, including backward gradients. |
