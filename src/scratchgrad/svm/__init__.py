"""Support vector machines.

``LinearSVM`` fits the soft-margin primal objective directly by subgradient
descent — no kernel trick. A kernelised ``SVC``/SMO solver is explicitly
optional per ``plan.md`` §7 and is not implemented here.
"""

from scratchgrad.svm.linear_svm import LinearSVM

__all__ = ["LinearSVM"]
