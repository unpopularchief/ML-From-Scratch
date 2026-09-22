# Recurrent cells and backpropagation through time

This is the finalized derivation for M4's recurrent unit. The implementation
keeps recurrence explicit: inputs are batch-first, each timestep is evaluated
in a Python loop, and backward traverses the cached states in reverse.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $(N,T,D)$ | batch-first input sequence |
| $x_t$ | $(N,D)$ | input at timestep $t$ |
| $h_t$ | $(N,H)$ | hidden state at timestep $t$ |
| $c_t$ | $(N,H)$ | LSTM cell state at timestep $t$ |
| $Y$ | $(N,T,H)$ | every hidden state, stacked over time |

Both sequence layers accept optional initial states; omitted states are zero.
They return $Y$, retain final states as attributes, and expose the gradients
of supplied initial states after backward. This preserves the project's
ordinary Module composition contract: forward returns one array and backward
returns $dX$.

## 1. Vanilla RNN cell

A tanh RNN reuses the same input matrix, recurrent matrix, and bias at every
timestep:

$$
a_t=x_tW_{ih}+h_{t-1}W_{hh}+b,
\qquad
h_t=\tanh(a_t).
$$

Here $W_{ih}\in\mathbb{R}^{D\times H}$,
$W_{hh}\in\mathbb{R}^{H\times H}$, and $b\in\mathbb{R}^{H}$. RNNCell
evaluates one step. RNN repeatedly evaluates the same equation and returns
$Y=[h_1,\ldots,h_T]$.

## 2. RNN backpropagation through time

Let $\bar h_t^{out}=\partial L/\partial Y_t$ be the direct gradient arriving
from the layer above. A state also affects every later timestep, so its total
gradient includes the gradient flowing backward through the recurrence:

$$
\bar h_t=\bar h_t^{out}+\bar h_t^{future}.
$$

Since $\partial\tanh(a_t)/\partial a_t=1-h_t^2$,

$$
\bar a_t=\bar h_t\odot(1-h_t^2).
$$

The remaining derivatives are ordinary affine-layer derivatives:

$$
\frac{\partial L}{\partial x_t}=\bar a_tW_{ih}^{\mathsf T},
\qquad
\bar h_{t-1}^{future}=\bar a_tW_{hh}^{\mathsf T},
$$

$$
\frac{\partial L}{\partial W_{ih}}
  \mathrel{+}=x_t^{\mathsf T}\bar a_t,
\qquad
\frac{\partial L}{\partial W_{hh}}
  \mathrel{+}=h_{t-1}^{\mathsf T}\bar a_t,
\qquad
\frac{\partial L}{\partial b}
  \mathrel{+}=\sum_n\bar a_{t,n}.
$$

Backward iterates from $T$ to $1$. The += is essential: the weights are
shared across time, so their gradient is the sum of their contribution at
every use. An optional terminal gradient is added to the final timestep's
gradient rather than replacing it.

Repeated multiplication by $W_{hh}$ and by tanh derivatives below one
explains the vanilla RNN's vanishing-gradient problem; singular values above
one can instead produce exploding gradients.

## 3. LSTM forward pass

An LSTM introduces a cell state with an additive update. One matrix operation
produces four gate preactivations in PyTorch-compatible i, f, g, o order:

$$
A_t=x_tW_{ih}+h_{t-1}W_{hh}+b
    =[a_{i,t},a_{f,t},a_{g,t},a_{o,t}],
$$

where the matrices have shapes $(D,4H)$ and $(H,4H)$. The gates are

$$
i_t=\sigma(a_{i,t}),\qquad
f_t=\sigma(a_{f,t}),\qquad
g_t=\tanh(a_{g,t}),\qquad
o_t=\sigma(a_{o,t}).
$$

The cell and hidden updates are

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,
\qquad
h_t=o_t\odot\tanh(c_t).
$$

The forget gate chooses how much old memory survives, the input gate chooses
how much candidate memory is written, and the output gate chooses how much
of the cell is exposed as the hidden state.

## 4. LSTM backpropagation through time

Reverse traversal carries two recurrent gradients. First combine the direct
hidden-output gradient with the future hidden gradient:

$$
\bar h_t=\bar h_t^{out}+\bar h_t^{future}.
$$

The hidden state contributes to the output gate and to the current cell:

$$
\bar o_t=\bar h_t\odot\tanh(c_t),
$$

$$
\bar c_t=\bar c_t^{future}
 +\bar h_t\odot o_t\odot(1-\tanh^2(c_t)).
$$

Differentiate the additive cell update:

$$
\bar f_t=\bar c_t\odot c_{t-1},\qquad
\bar c_{t-1}^{future}=\bar c_t\odot f_t,
$$

$$
\bar i_t=\bar c_t\odot g_t,\qquad
\bar g_t=\bar c_t\odot i_t.
$$

Then differentiate each gate nonlinearity:

$$
\bar a_{i,t}=\bar i_t\odot i_t(1-i_t),\qquad
\bar a_{f,t}=\bar f_t\odot f_t(1-f_t),
$$

$$
\bar a_{g,t}=\bar g_t\odot(1-g_t^2),\qquad
\bar a_{o,t}=\bar o_t\odot o_t(1-o_t).
$$

Concatenate these four arrays into $\bar A_t$ in the same i, f, g, o order.
The input, previous-hidden, and parameter gradients then use the same affine
derivatives as the vanilla RNN, replacing $\bar a_t$ with $\bar A_t$.

The route $\bar c_{t-1}^{future}=\bar c_t\odot f_t$ avoids a mandatory
matrix multiplication and tanh derivative at every step. When the forget
gate stays near one, the LSTM can therefore carry gradients over much longer
intervals than a vanilla RNN.

## 5. API and initialization

- RNNCell and LSTMCell expose one timestep and return previous-state
  gradients directly from backward.
- RNN and LSTM consume only batch-first $(N,T,D)$ sequences, always return
  every hidden state, and implement full rather than truncated BPTT.
- RNN.h_n and LSTM.h_n/c_n retain final states. grad_h0/grad_c0 retain
  initial-state gradients after backward.
- The implementation is single-layer and unidirectional. Packed sequences,
  variable-length masks, dropout, projections, peepholes, bidirectionality,
  and truncated BPTT are outside this unit.
- Input and recurrent matrices are independently Xavier-initialized by
  default, appropriate for tanh/sigmoid gates; zero initialization is also
  available for deterministic analytic examples. Biases start at zero.
- A single bias is used because separate input and recurrent biases are
  algebraically redundant. PyTorch parity maps it to bias_ih and sets
  bias_hh to zero.

## 6. Testing plan

| Tier | Check |
| --- | --- |
| Analytic | Direct one-step equations, gate behavior at zero preactivation, state/output shapes, and a hand-computed reverse-time accumulation. |
| Gradient check | Central finite differences through complete unrolled RNN/LSTM graphs for inputs, both weight matrices, bias, initial hidden/cell states, and separate final-state loss edges. |
| Contract | Invalid sizes/shapes, non-empty batch-first sequences, bias-disabled parameter lists, initialization, and parameter/gradient ordering. |
| Reference | Identical inputs, states, parameters, output gradients, and terminal-state gradients compared with PyTorch cells and single-layer sequence modules. |
