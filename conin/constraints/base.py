from abc import ABC


class Constraint(ABC):
    """
    Abstract base class for all conin constraints.

    A constraint is a plain container.  Concrete subclasses hold the user
    function (``func``), a human-readable ``name``, the number of arguments
    accepted by ``func`` (``num_args``) and any class-specific metadata.
    Applying a constraint to a model is the job of the inference and
    representation code, not of the constraint itself.
    """

    family = None
    """str or None: The constraint family.

    A constrained model requires all of its constraints to belong to a single
    family.  Most constraint classes form their own family (e.g. ``"pyomo"``,
    ``"toulbar2"``, ``"oracle"``).  Concrete subclasses must override this;
    the base class value is ``None``.
    """
