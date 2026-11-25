from abc import ABC

# ConstrainedGraphicalModel.  It is in a layer above this one, so it registers
# itself here (conin/constrained_models/base.py).
_constrained_class = None


def set_constrained_class(cls):
    """Set the class that :meth:`GraphicalModel.representation` wraps models in."""
    global _constrained_class
    _constrained_class = cls


class GraphicalModel(ABC):
    """Base class for unconstrained graphical models.

    Every unconstrained model class in :mod:`conin.graphical_models` derives
    from this class, so ``isinstance(model, GraphicalModel)`` identifies them.
    """

    __slots__ = ()

    def representation(self, kind, **structure_params):
        """Build the ``kind`` representation of this model without constraints.

        The same as ``ConstrainedGraphicalModel(model).representation(kind)``.
        A new representation is built on every call.

        Raises
        ------
        InvalidInputError
            If no builder is registered for ``(kind, model_type, None)``.
        TypeError
            If no representation builder is registered for this model's class
            or one of its base classes.
        """
        return _constrained_class(self).representation(kind, **structure_params)


def node_sort_key(node):
    """Sort key of a node: a name ``x`` sorts as ``(x,)``, so names and ``(name, t)`` tuples mix."""
    return node if isinstance(node, tuple) else (node,)
