"""Representation ABC and the builder registry.

A *representation* is whatever a backend needs to work on a constrained model:
a wrapped oracle, a Pyomo formulation, aligned MVRs, and so on.  It is built
from a :class:`~conin.constrained_models.ConstrainedGraphicalModel` by a
*builder* that is registered under ``(kind, model_type, constraint_family)``.

The registry is the single source of truth for which (kind, family) pairs a
model type supports.
"""

import importlib
from abc import ABC

from conin.exceptions import InvalidInputError


class Representation(ABC):
    """Base class for representations built from a constrained model.

    ``cmodel.representation(kind, ...)`` builds a new representation on every
    call (nothing is cached), so the caller owns what it gets.
    """

    kind = None
    """str or None: The representation kind this class is registered under."""


# (kind, model_type, family) -> builder(cmodel, **structure_params) -> Representation
_REGISTRY = {}

# Modules that register the built-in builders.  They are imported on first use
# of the registry, not at ``import conin`` time: they import backend modules
# (e.g. conin.hidden_markov_model) that themselves import the constrained models.
_BUILTIN_MODULES = (
    "conin.representations.oracle.hmm",
    "conin.representations.pyomo.mn",
    "conin.representations.pyomo.bn",
    "conin.representations.pyomo.dbn",
    "conin.representations.pyomo.hmm",
    "conin.representations.pyomo.hmm_network_flow",
    "conin.representations.toulbar2.mn",
    "conin.representations.toulbar2.bn",
    "conin.representations.toulbar2.dbn",
    "conin.representations.toulbar2.hmm",
    "conin.representations.factor.mn",
    "conin.representations.factor.bn",
    "conin.representations.factor.dbn",
    "conin.representations.factor.hmm",
    "conin.representations.mvr.hmm",
)
_builtins_loaded = False


def _load_builtins():
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        for name in _BUILTIN_MODULES:
            importlib.import_module(name)


def register_representation(kind, model_type, families=(None,)):
    """Decorator that registers a representation builder.

    The decorated builder is called as ``builder(cmodel, **structure_params)``
    and must return a :class:`Representation`.

    Parameters
    ----------
    kind : str
        Representation kind, e.g. ``"oracle"`` or ``"pyomo_network_flow"``.
    model_type : type
        The unconstrained graphical model class (``cmodel.model_type``).
    families : iterable of (str or None), optional
        Constraint families the builder accepts.  ``None`` stands for a model
        without constraints.

    Raises
    ------
    ValueError
        If a builder is already registered for one of the keys.
    """
    families = tuple(families)

    def decorator(builder):
        for family in families:
            key = (kind, model_type, family)
            if key in _REGISTRY and _REGISTRY[key] is not builder:
                raise ValueError(
                    f"A representation builder is already registered for {key}"
                )
        for family in families:
            _REGISTRY[(kind, model_type, family)] = builder
        return builder

    return decorator


def unregister_representation(kind, model_type, families=(None,)):
    """Remove registered builders (mainly for tests).  Missing keys are ignored."""
    for family in families:
        _REGISTRY.pop((kind, model_type, family), None)


def supported_representations(model_type):
    """Return the sorted ``(kind, family)`` pairs registered for ``model_type``.

    ``family`` is ``None`` for "no constraints".
    """
    _load_builtins()
    pairs = {(k, f) for (k, mt, f) in _REGISTRY if mt is model_type}
    return sorted(pairs, key=lambda p: (p[0], "" if p[1] is None else p[1]))


def get_representation_builder(kind, model_type, family):
    """Return the builder for ``(kind, model_type, family)``, or ``None``."""
    _load_builtins()
    return _REGISTRY.get((kind, model_type, family))


def build_representation(cmodel, kind, **structure_params):
    """Build the ``kind`` representation of ``cmodel``.

    Most callers should use ``cmodel.representation(kind, ...)``, which calls
    this.

    Raises
    ------
    InvalidInputError
        If no builder is registered for ``(kind, cmodel.model_type,
        cmodel.constraint_family)``.  The message lists the supported pairs.
    """
    model_type = cmodel.model_type
    family = cmodel.constraint_family
    builder = get_representation_builder(kind, model_type, family)
    if builder is None:
        supported = ", ".join(
            f"({k!r}, {f!r})" for k, f in supported_representations(model_type)
        )
        raise InvalidInputError(
            f"No {kind!r} representation is available for "
            f"{type(cmodel).__name__} with constraint family {family!r}. "
            f"Supported (kind, family) pairs: {supported or 'none'}."
        )
    return builder(cmodel, **structure_params)
