"""The ``model.V(...)`` variable accessor shared by the constraint backends.

It lives in :mod:`conin.util` (not in :mod:`conin.representations.pyomo`) so
that :mod:`conin.constraints.algebraic` can use it without importing the
representations layer, which itself imports :mod:`conin.constraints`.  The
public name is :class:`conin.representations.pyomo.PyomoVarWrapper`.
"""

import pprint

from .misc import State


class PyomoVarWrapper(dict):
    """Map ``(node, State(value))`` to a Pyomo variable; callable as ``V(node, value)``.

    ``V(r, s)`` looks up ``(r, State(s))`` and ``V(r, t, s)`` looks up
    ``((r, t), State(s))`` (the time-indexed nodes of DBNs and HMMs).

    Raises
    ------
    ValueError
        If called with other than 2 or 3 arguments.
    KeyError
        If the variable does not exist.
    """

    def __init__(self, *arg, **kw):
        super(PyomoVarWrapper, self).__init__(*arg, **kw)

    def pprint(self):  # pragma:nocover
        pprint.pprint(self)

    def __call__(self, *args):
        if len(args) == 2:
            r, s = args
        elif len(args) == 3:
            r, i, s = args
            r = (r, i)
        else:
            raise ValueError("There must be either 2 or 3 arguments")

        if type(s) is not State:
            s = State(s)
        return dict.__getitem__(self, (r, s))
