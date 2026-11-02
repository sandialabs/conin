"""Read-only lists and dicts for the data of immutable models.

The graphical models are immutable (see ``doc/sphinx/architecture.rst``), but
their nodes, states, factor values and CPD values are lists and dicts.
:func:`freeze` copies such data into :class:`FrozenList` and
:class:`FrozenDict`, which raise ``TypeError`` on every mutating method and
otherwise behave as the built-in types: they are ``list``/``dict`` instances,
compare equal to lists and dicts with the same items, and print the same.
"""


def _read_only(self, *args, **kwds):
    raise TypeError(
        f"{type(self).__name__} is read-only: the data of a CONIN model cannot "
        "be changed (copy it with list() or dict() first)"
    )


class FrozenList(list):
    """A ``list`` whose items cannot be changed."""

    __slots__ = ()

    __setitem__ = __delitem__ = __iadd__ = __imul__ = _read_only
    append = extend = insert = pop = remove = clear = sort = reverse = _read_only

    def __reduce__(self):
        # pickle and copy rebuild the list from its items (list.__reduce_ex__
        # would append the items one by one, which is read-only here).
        return (type(self), (list(self),))

    def __repr__(self):
        return list.__repr__(self)


class FrozenDict(dict):
    """A ``dict`` whose items cannot be changed."""

    __slots__ = ()

    __setitem__ = __delitem__ = __ior__ = _read_only
    pop = popitem = clear = update = setdefault = _read_only

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __repr__(self):
        return dict.__repr__(self)


def freeze(value):
    """Return a read-only copy of ``value``.

    Lists become :class:`FrozenList` and dicts :class:`FrozenDict`, with their
    items frozen recursively; tuples are rebuilt with frozen items.  Other
    values (states, probabilities, ...) are returned as they are.
    """
    if isinstance(value, (FrozenList, FrozenDict)):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(freeze(item) for item in value)
    return value
