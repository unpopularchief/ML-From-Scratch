"""First-order optimizers.

Generic update rules for minimizing a differentiable ``L(theta)`` given
only its gradient at each step -- unlike every other package in this
project, these have no objective/loss of their own; they consume a
gradient computed elsewhere (a classical estimator's own loss gradient
today; a hand-derived backward pass from ``nn.Module`` onward; ``autograd``
from M5 onward).

No shared base class -- see ``docs/derivations/optim.md`` section 7 for
why the five ``step`` bodies differ enough that one would not fit.
"""

from scratchgrad.optim.adam import Adam
from scratchgrad.optim.momentum import Momentum
from scratchgrad.optim.nesterov import Nesterov
from scratchgrad.optim.rmsprop import RMSprop
from scratchgrad.optim.sgd import SGD

__all__ = ["Adam", "Momentum", "Nesterov", "RMSprop", "SGD"]
