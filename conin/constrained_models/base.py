"""Base class for constrained graphical models."""

from conin.constraints.base import Constraint
from conin.exceptions import InvalidInputError
from conin.representations.base import (
    build_representation,
    supported_representations,
)


class ConstrainedGraphicalModel:
    """An immutable graphical model paired with an immutable tuple of constraints.

    Parameters
    ----------
    model : GraphicalModel
        The unconstrained model. It must be an instance of ``model_type``.
    constraints : Constraint or iterable of Constraint, optional
        The constraints: a single constraint, or a list, tuple or other
        iterable of constraints. They must all belong to one constraint family
        (``type(c).family``), and the family must be supported for
        ``model_type``: some representation builder must be registered for it
        (see :func:`conin.representations.supported_representations`).
        ``None`` is treated as no constraints.
    data : object, optional
        Application data passed to the representation builders (e.g. to MVR
        factories or Pyomo constraint functions).

    Notes
    -----
    Instances are immutable: assigning or deleting attributes raises
    ``AttributeError``. Use :meth:`with_constraints` or :meth:`with_model` to
    derive a new object.

    Representations are built by :meth:`representation` on every call (they
    are not cached).
    """

    __slots__ = ("_model", "_constraints", "_family", "_data")

    model_type = None
    """type: The unconstrained model class this wrapper accepts (set by subclasses)."""

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _normalise_constraints(self, constraints):
        if constraints is None:
            return ()
        if isinstance(constraints, Constraint):
            return (constraints,)
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
        if family is not None:
            # A family is accepted iff some representation builder is registered
            # for (model_type, family): the registry is the single record of
            # what is supported.
            supported = self._supported_families()
            if family not in supported:
                raise InvalidInputError(
                    f"{type(self).__name__} does not support constraint family "
                    f"{family!r}; supported families: {supported}"
                )
        return family

    @classmethod
    def _supported_families(cls):
        """The sorted families of the builders registered for ``model_type``."""
        return sorted(
            {f for _, f in supported_representations(cls.model_type) if f is not None}
        )

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
        """Build the ``kind`` representation of this model.

        Calls ``builder(self, **structure_params)``. A new representation is
        built on every call; nothing is cached, so the caller owns the result.

        Raises
        ------
        InvalidInputError
            If no builder is registered for ``(kind, model_type,
            constraint_family)``. The message lists the supported pairs.
        """
        return build_representation(self, kind, **structure_params)

    @classmethod
    def ensure(cls, model):
        """Return ``model`` if it is already a ``cls``, else ``cls(model)``.

        Lets functions that accept either a bare or a constrained model work
        on the constrained one. A bare model is wrapped with no constraints.
        Anything else raises ``InvalidInputError``, as the constructor does.
        """
        if isinstance(model, cls):
            return model
        return cls(model)

    def with_constraints(self, constraints):
        """Return a new object with the same model and data and new constraints.

        ``constraints`` is a single constraint or an iterable of constraints,
        as in the constructor.
        """
        return type(self)(self._model, constraints, data=self._data)

    def with_model(self, model):
        """Return a new object with the same constraints and data and a new model."""
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
        return (self._model, self._constraints, self._family, self._data)

    def __setstate__(self, state):
        model, constraints, family, data = state
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_constraints", constraints)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_data", data)

    def __repr__(self):
        names = [getattr(c, "name", type(c).__name__) for c in self._constraints]
        return (
            f"{type(self).__name__}(model={type(self._model).__name__}, "
            f"constraint_family={self._family!r}, constraints={names!r})"
        )
