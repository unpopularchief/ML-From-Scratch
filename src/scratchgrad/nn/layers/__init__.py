"""Parameterized layers. Only ``Linear`` this unit -- see plan.md section 1.

``conv2d``, ``pooling``, ``flatten``, ``dropout``, ``batchnorm``, ``rnn``,
``lstm`` arrive in their own later units (M3's Dropout/BatchNorm, then M4).
"""

from scratchgrad.nn.layers.linear import Linear

__all__ = ["Linear"]
