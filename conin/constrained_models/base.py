"""Base class for constrained graphical models."""

from conin.constraints.base import Constraint
from conin.exceptions import InvalidInputError


def _freeze(value):
    """Turn a structure parameter into a hashable cache-key component."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    hash(value)  # raises TypeError for other unhashable values
    return value


class ConstrainedGraphicalModel:
    """An immutable graphical model paired with an immutable tuple of constraints.

    Parameters
    ----------
    model : GraphicalModel
        The unconstrained model. It must be an instance of ``model_type``.
    constraints : iterable of Constraint, optional
        The constraints (a list, tuple or other iterable, not a single
        constraint). They must all belong to one constraint family
        (``type(c).family``). ``None`` is treated as no constraints.
    data : object, optional
        Application data passed to the representation builders (e.g. to MVR
        factories or Pyomo constraint functions). It is not part of the
        representation cache key.

    Notes
    -----
    Instances are immutable: assigning or deleting attributes raises
    ``AttributeError``. Use :meth:`with_constraints` or :meth:`with_model` to
    derive a new object.

    Representations are built lazily by :meth:`representation`, cached on the
    object, and handed out as ``repn.get_copy()``.
    """

    __slots__ = ("_model", "_constraints", "_family", "_data", "_cache")

    model_type = None
    """type: The unconstrained model class this wrapper accepts (set by subclasses)."""

    supported_families = None
    """tuple or None: Constraint families accepted at construction (``None`` = any)."""

    def __init__(self, model, constraints=(), *, data=None):
        if self.model_type is None:
            raise TypeError(
                f"{type(self).__name__} does not define model_type; "
                "instantiate one of the concrete constrained model classes"
            )
        if not isinstance(model, self.model_type):
            raise InvalidInputError(
                f"{type(self).__name__} expects a {self.model_type.__name__} "
                f"as its model, got {type(model).__name__}"
            )
        constraints = self._normalise_constraints(constraints)
        family = self._derive_family(constraints)

        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_constraints", constraints)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_cache", {})

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _normalise_constraints(self, constraints):
        if constraints is None:
            return ()
        if isinstance(constraints, Constraint):
            raise InvalidInputError(
                f"{type(self).__name__}: constraints must be an iterable of "
                f"Constraint objects, not a single {type(constraints).__name__}; "
                "wrap it in a list"
            )
        if isinstance(constraints, (str, bytes)):
            raise InvalidInputError(
                f"{type(self).__name__}: constraints must be an iterable of "
                f"Constraint objects, got {type(constraints).__name__}"
            )
        try:
            constraints = tuple(constraints)
        except TypeError:
            raise InvalidInputError(
                f"{type(self).__name__}: constraints must be an iterable of "
                f"Constraint objects, got {type(constraints).__name__}"
            ) from None
        for i, c in enumerate(constraints):
            if not isinstance(c, Constraint):
                raise InvalidInputError(
                    f"{type(self).__name__}: constraint {i} is not a "
                    f"conin.constraints.Constraint: {type(c).__name__}"
                )
        return constraints

    def _derive_family(self, constraints):
        families = []
        for c in constraints:
            f = type(c).family
            if f not in families:
                families.append(f)
        if len(families) > 1:
            raise InvalidInputError(
                f"{type(self).__name__}: all constraints must belong to one "
                f"constraint family, found {sorted(map(str, families))}"
            )
        family = families[0] if families else None
        if (
            family is not None
            and self.supported_families is not None
            and family not in self.supported_families
        ):
            raise InvalidInputError(
                f"{type(self).__name__} does not support constraint family "
                f"{family!r}; supported families: "
                f"{[f for f in self.supported_families if f is not None]}"
            )
        return family

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The wrapped unconstrained graphical model."""
        return self._model

    @property
    def constraints(self):
        """tuple of Constraint: The constraints (possibly empty)."""
        return self._constraints

    @property
    def constraint_family(self):
        """str or None: The family shared by all constraints (``None`` if there are none)."""
        return self._family

    @property
    def data(self):
        """The application data passed to the representation builders (or ``None``)."""
        return self._data

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def representation(self, kind, **structure_params):
        """Return a copy of the ``kind`` representation of this model.

        The representation is built on first use with
        ``builder(self, **structure_params)`` and cached under
        ``(kind, structure_params)``. Every call returns ``repn.get_copy()``.

        Raises
        ------
        InvalidInputError
            If no builder is registered for ``(kind, model_type,
            constraint_family)``. The message lists the supported pairs.
        """
        from conin.representations.base import build_representation

        key = (kind, _freeze(structure_params))
        repn = self._cache.get(key)
        if repn is None:
            repn = build_representation(self, kind, **structure_params)
            self._cache[key] = repn
        return repn.get_copy()

    def with_constraints(self, constraints):
        """Return a new object with the same model and data and new constraints.

        The new object starts with an empty representation cache.
        """
        return type(self)(self._model, constraints, data=self._data)

    def with_model(self, model):
        """Return a new object with the same constraints and data and a new model.

        The new object starts with an empty representation cache.
        """
        return type(self)(model, self._constraints, data=self._data)

    # ------------------------------------------------------------------
    # Immutability, copying and pickling
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot set {name!r}. "
            "Use with_constraints() or with_model() to derive a new object."
        )

    def __delattr__(self, name):
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete {name!r}"
        )

    def __getstate__(self):
        # The representation cache is not copied or pickled; copies rebuild lazily.
        return (self._model, self._constraints, self._family, self._data)

    def __setstate__(self, state):
        model, constraints, family, data = state
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_constraints", constraints)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_cache", {})

    def __repr__(self):
        names = [getattr(c, "name", type(c).__name__) for c in self._constraints]
        return (
            f"{type(self).__name__}(model={type(self._model).__name__}, "
            f"constraint_family={self._family!r}, constraints={names!r})"
        )
