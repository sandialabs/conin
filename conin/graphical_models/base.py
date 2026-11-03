from abc import ABC

# The constrained model class of each model class (model_type -> class).  The
# constrained models are in a layer above this one, so they register
# themselves (ConstrainedGraphicalModel.__init_subclass__).
_CONSTRAINED_CLASSES = {}


def register_constrained_class(cls):
    """Register ``cls`` as the constrained model class of ``cls.model_type``.

    Called by ``ConstrainedGraphicalModel.__init_subclass__`` for every class
    that sets ``model_type`` itself.

    Raises
    ------
    TypeError
        If another class is already registered for ``cls.model_type``.
    """
    registered = _CONSTRAINED_CLASSES.setdefault(cls.model_type, cls)
    if registered is not cls:
        raise TypeError(
            f"{cls.__qualname__}: {registered.__qualname__} is already the "
            f"constrained model class of {cls.model_type.__name__}"
        )


def constrained_classes():
    """Return the registered constrained model classes, in registration order."""
    return list(_CONSTRAINED_CLASSES.values())


def constrained_class(model_type):
    """Return the constrained model class of ``model_type`` (or ``None``).

    A subclass of a model class gets the constrained class of its nearest
    registered base class.
    """
    for base in model_type.__mro__:
        if base in _CONSTRAINED_CLASSES:
            return _CONSTRAINED_CLASSES[base]
    return None


class GraphicalModel(ABC):
    """Base class for unconstrained graphical models.

    Every unconstrained model class in :mod:`conin.graphical_models` derives
    from this class, so ``isinstance(model, GraphicalModel)`` identifies them.
    """

    __slots__ = ()

    def representation(self, kind, **structure_params):
        """Build the ``kind`` representation of this model without constraints.

        The same as wrapping the model in its constrained model class with no
        constraints, e.g. ``ConstrainedHiddenMarkovModel(hmm).representation(kind)``.
        A new representation is built on every call.

        Raises
        ------
        InvalidInputError
            If no builder is registered for ``(kind, type(self), None)``.
        TypeError
            If no constrained model class is registered for this model's
            class.
        """
        cls = constrained_class(type(self))
        if cls is None:
            raise TypeError(
                f"No constrained model class is registered for "
                f"{type(self).__name__}"
            )
        return cls(self).representation(kind, **structure_params)


def node_sort_key(node):
    """Sort key of a node: a name ``x`` sorts as ``(x,)``, so names and ``(name, t)`` tuples mix."""
    return node if isinstance(node, tuple) else (node,)
