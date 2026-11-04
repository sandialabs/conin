import copy
from abc import ABC

from conin.util.frozen import freeze


class Constraint(ABC):
    """
    Abstract base class for all conin constraints.

    A constraint is a plain container.  Concrete subclasses hold the user
    function (``func``), a human-readable ``name``, the number of arguments
    accepted by ``func`` (``num_args``) and any class-specific metadata.
    Applying a constraint to a model is the job of the inference and
    representation code, not of the constraint itself.

    Constraints are immutable, like the models and the constrained models:
    assigning or deleting an attribute raises ``AttributeError``, and lists
    and dicts are stored as read-only copies (:func:`conin.util.frozen.freeze`).
    Subclasses set their attributes in ``__init__`` with :meth:`_set`, and
    derive changed constraints with :meth:`_replace`.
    """

    family = None
    """str or None: The constraint family.

    A constrained model requires all of its constraints to belong to a single
    family.  Most constraint classes form their own family (e.g. ``"pyomo"``,
    ``"toulbar2"``, ``"oracle"``).  Concrete subclasses must override this;
    the base class value is ``None``.
    """

    def _set(self, **attributes):
        """Set attributes, as read-only copies; only for ``__init__`` and :meth:`_replace`."""
        for name, value in attributes.items():
            object.__setattr__(self, name, freeze(value))

    def _replace(self, **attributes):
        """Return a shallow copy of this constraint with other ``attributes``."""
        result = copy.copy(self)
        result._set(**attributes)
        return result

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __delattr__(self, name):
        raise AttributeError(
            f"{type(self).__name__} is immutable; cannot delete {name!r}"
        )


def constraint_families(constraints):
    """The set of distinct families of ``constraints``."""
    return {type(c).family for c in constraints}
