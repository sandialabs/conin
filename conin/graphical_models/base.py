from abc import ABC


class GraphicalModel(ABC):
    """Marker base class for unconstrained graphical models.

    Every unconstrained model class in :mod:`conin.graphical_models` derives
    from this class, so ``isinstance(model, GraphicalModel)`` identifies them.
    The class defines no behaviour.
    """

    __slots__ = ()


def node_sort_key(node):
    """Sort key of a node: a name ``x`` sorts as ``(x,)``, so names and ``(name, t)`` tuples mix."""
    return node if isinstance(node, tuple) else (node,)
