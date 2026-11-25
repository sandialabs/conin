"""Representation ABC and the builder registry.

A *representation* is whatever a backend needs to work on a constrained model:
a wrapped oracle, a Pyomo formulation, aligned MVRs, and so on.  It is built
from a :class:`~conin.constrained_models.ConstrainedGraphicalModel` by a
*builder* that is registered under ``(kind, model_type, constraint_family)``.

The registry is the single source of truth for which (kind, family) pairs a
model type supports.  A constrained model with constraints of several
families can use a kind if one builder is registered for all of them.
"""

import importlib
from abc import ABC

from conin.constraints.base import group_by_family
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
# of the registry, not at ``import conin`` time: they import modules (e.g.
# conin.io, through the toulbar2 representations) that themselves import the
# constrained models.
_BUILTIN_MODULES = (
    "conin.representations.matvec.hmm",
    "conin.representations.oracle.hmm",
    "conin.representations.recursive.hmm",
    "conin.representations.pyomo.mn",
    "conin.representations.pyomo.bn",
    "conin.representations.pyomo.dbn",
    "conin.representations.pyomo.hmm",
    "conin.representations.pyomo.hmm_network_flow",
    "conin.representations.toulbar2.mn",
    "conin.representations.toulbar2.bn",
    "conin.representations.toulbar2.dbn",
    "conin.representations.toulbar2.hmm",
    "conin.representations.variable_elimination.mn",
    "conin.representations.variable_elimination.bn",
    "conin.representations.variable_elimination.dbn",
    "conin.representations.variable_elimination.hmm",
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
        Registering a builder for a new model class makes the class (and its
        subclasses) wrappable in a ``ConstrainedGraphicalModel``.
    families : iterable of (str or None), optional
        Constraint families the builder accepts.  ``None`` stands for a model
        without constraints.

    Raises
    ------
    InvalidInputError
        If a builder is already registered for one of the keys.
    """
    families = tuple(families)

    def decorator(builder):
        for family in families:
            key = (kind, model_type, family)
            if key in _REGISTRY and _REGISTRY[key] is not builder:
                raise InvalidInputError(
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


def registered_model_type(cls):
    """Return the nearest class in ``cls.__mro__`` with a registered builder.

    This is the ``model_type`` of a constrained model of a ``cls`` model (so a
    subclass of a model class uses the builders of that class).  ``None`` if
    no class in the MRO has one.
    """
    _load_builtins()
    registered = {mt for (_, mt, _) in _REGISTRY}
    for base in cls.__mro__:
        if base in registered:
            return base
    return None


def get_representation_builder(kind, model_type, family):
    """Return the builder for ``(kind, model_type, family)``, or ``None``."""
    _load_builtins()
    return _REGISTRY.get((kind, model_type, family))


def _builder_for_families(kind, model_type, families):
    """The builder registered for ``kind`` and all of ``families``, or ``None``.

    ``families`` empty means no constraints (family ``None``).
    """
    if not families:
        return get_representation_builder(kind, model_type, None)
    builders = {get_representation_builder(kind, model_type, f) for f in families}
    if len(builders) == 1 and None not in builders:
        return builders.pop()
    return None


def representation_kinds(model_type, families):
    """The sorted kinds with a builder for ``model_type`` and all of ``families``."""
    _load_builtins()
    kinds = {k for (k, mt, _) in _REGISTRY if mt is model_type}
    return sorted(
        k for k in kinds if _builder_for_families(k, model_type, families) is not None
    )


def combinable_families(model_type):
    """``{kind: families}`` for the kinds of ``model_type`` that accept several families.

    ``families`` is the sorted tuple of the families that the kind's builder
    accepts together.
    """
    _load_builtins()
    by_builder = {}
    for (k, mt, f), builder in _REGISTRY.items():
        if mt is model_type and f is not None:
            by_builder.setdefault((k, builder), set()).add(f)
    return {
        k: tuple(sorted(fams))
        for (k, _), fams in sorted(by_builder.items(), key=lambda kv: kv[0][0])
        if len(fams) > 1
    }


def build_representation(cmodel, kind, **structure_params):
    """Build the ``kind`` representation of ``cmodel``.

    Most callers should use ``cmodel.representation(kind, ...)``, which calls
    this.

    Raises
    ------
    InvalidInputError
        If no builder is registered for ``kind``, ``cmodel.model_type`` and
        all of ``cmodel.constraint_families``.  The message lists the
        supported pairs.
    """
    model_type = cmodel.model_type
    families = cmodel.constraint_families
    builder = _builder_for_families(kind, model_type, families)
    if builder is None:
        supported = ", ".join(
            f"({k!r}, {f!r})" for k, f in supported_representations(model_type)
        )
        if len(families) > 1:
            what = f"constraint families {list(families)}"
        else:
            what = f"constraint family {families[0] if families else None!r}"
        raise InvalidInputError(
            f"No {kind!r} representation is available for "
            f"{model_type.__name__} with {what}. "
            f"Supported (kind, family) pairs: {supported or 'none'}."
        )
    return builder(cmodel, **structure_params)


def apply_constraints(constraints, appliers, result, *, backend):
    """Apply ``constraints`` family by family with the appliers of their families.

    This is the dispatch shared by the backend appliers
    (``apply_pyomo_constraints``, ``apply_toulbar2_constraints``,
    ``apply_variable_elimination_constraints``): each one passes a table from
    family to the function that applies the constraints of that family.  The
    families are applied in the order in which they first appear, each to
    the result of the previous one.

    Parameters
    ----------
    constraints : sequence of Constraint
        The constraints, of one or several families.
    appliers : dict
        ``{family: apply(constraints, result) -> result}``.
    result : object
        What the first applier gets, e.g. the model to add constraints to.
    backend : str
        Named in the error messages.

    Returns
    -------
    object
        What the last applier returns; ``result`` for no constraints.

    Raises
    ------
    TypeError
        For constraints of a family that ``appliers`` does not handle (then
        no applier is called).
    """
    groups = group_by_family(constraints)
    for family, cons in groups.items():
        if family not in appliers:
            raise TypeError(
                f"The {backend} backend does not apply constraints of family "
                f"{family!r} ({type(cons[0]).__name__}); supported families: "
                f"{sorted(appliers)}"
            )
    for family, cons in groups.items():
        result = appliers[family](cons, result)
    return result
