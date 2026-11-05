"""Normalise the model argument of a query to a constrained CONIN model.

Internal: the query entry points (e.g. ``map_query``) call
:func:`as_constrained_model` first, so that every query runs on a constrained
model (a bare model is wrapped with no constraints).
"""

from conin.graphical_models.base import constrained_class, constrained_classes

from .base import ConstrainedGraphicalModel


def as_constrained_model(pgm):
    """Return ``pgm`` as a constrained CONIN model.

    A constrained model is returned as it is, and bare CONIN models are
    wrapped with no constraints.

    Raises
    ------
    TypeError
        If ``pgm`` is not a supported model.
    """
    if isinstance(pgm, ConstrainedGraphicalModel):
        return pgm
    cls = constrained_class(type(pgm))
    if cls is not None:
        return cls(pgm)
    expected = []
    for cls in constrained_classes():
        expected += [cls.model_type.__name__, cls.__name__]
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. Expected one of: "
        f"{', '.join(expected)}."
    )
