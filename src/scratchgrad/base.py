"""The shared estimator contract.

Deliberately scikit-learn-shaped (see plan.md section 1): hyperparameters
are constructor arguments and are never touched by ``fit``; learned
attributes get a trailing underscore (``coef_``, ``labels_``); randomness
always flows through an explicit ``random_state``. ``Estimator`` itself
stays deliberately small — introspection and a ``repr``, nothing else. No
cloning, no meta-estimator machinery: those solve problems this project
doesn't have.
"""

from __future__ import annotations

import inspect


class Estimator:
    """Base class providing ``get_params``/``__repr__`` for every model.

    Subclasses implement ``fit`` (and ``predict``, ``transform``, etc. as
    appropriate) themselves — this base class does not declare abstract
    methods for them, since the right signature differs between a
    regressor, a classifier, and a transformer. What it standardizes is
    purely mechanical: reading back the hyperparameters a subclass was
    constructed with.

    Notes
    -----
    ``get_params`` inspects ``__init__``'s signature rather than requiring
    subclasses to register their parameters some other way, so adding a
    new hyperparameter to a subclass's ``__init__`` is the only step
    needed to have it picked up here too.

    """

    def get_params(self) -> dict[str, object]:
        """Return the constructor arguments this instance was created with.

        Returns
        -------
        dict[str, object]
            Maps each named ``__init__`` parameter (other than ``self``) to
            its current value on this instance. Empty for a subclass with
            no ``__init__`` of its own — inspecting the inherited
            ``object.__init__`` would otherwise report a bogus
            ``(*args, **kwargs)`` signature, so ``*args``/``**kwargs``
            entries are filtered out rather than treated as parameters.

        """
        signature = inspect.signature(self.__init__)
        return {
            name: getattr(self, name)
            for name, param in signature.parameters.items()
            if param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

    def __repr__(self) -> str:
        """Return ``ClassName(param=value, ...)`` from :meth:`get_params`."""
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"
