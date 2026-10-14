import inspect

from ..exceptions import InvalidInputError
from .base import Constraint
from .factor_oracle import FactorOracleConstraint


class OracleConstraint(Constraint):

    family = "oracle"

    def __init__(
        self,
        *,
        func=None,
        name=None,
        partial_func=None,
        same_partial_as_func=None,
    ):
        """
        Initialize an OracleConstraint object.

        An oracle constraint is a pure predicate.  The user function always
        receives a **dict** mapping keys to state values and returns
        ``True`` / ``False``.

        * For BN/MN the keys are node names, e.g. ``{"Dyspnoea": 1, "Xray": 0}``.
        * For DBN the keys are ``(name, t)`` tuples, e.g. ``{("A", 0): 0}``.
        * For HMM the keys are time indices, e.g. ``{0: "rainy", 1: "sunny"}``.

        For predicates with an explicit node scope that are materialised into
        a factor or CPD (e.g. for variable elimination), use
        :class:`~conin.constraints.FactorOracleConstraint` instead.

        Parameters
        ----------
        func : callable, optional
            The constraint predicate ``func(assignment) -> bool``, or
            ``func(assignment, data) -> bool`` for constraints that need
            inference-time data.
        name : str, optional
            Human-readable name.  Defaults to ``func.__name__`` when *func*
            is provided.
        partial_func : callable, optional
            A function ``(T, states_dict) -> bool`` that may prune partial
            sequences during A* search.  Only relevant for HMM oracle
            constraints.
        same_partial_as_func : bool, optional
            If ``True``, ``partial_func`` is set to
            ``lambda T, states: func(states)``.
        """
        self.func = func

        if same_partial_as_func is True:
            self.partial_func = lambda T, states: func(states)
        elif partial_func is not None:
            self.partial_func = partial_func
        else:
            self.partial_func = lambda T, states: True

        # Determine how many positional args `func` accepts (1 or 2).
        if func is not None:
            self.num_args = len(inspect.signature(func).parameters)
        else:
            self.num_args = None

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed constraint"  # Could also be none

    def __call__(self, *args, **kwargs):
        """
        Evaluate the predicate: ``constraint(states_dict) -> bool``.

        Raises
        ------
        InvalidInputError
            If the constraint function is not defined.
        """
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )

        return self.func(*args, **kwargs)


def oracle_constraint_fn(*, nodes=None, name=None, same_partial_as_func=None):
    """
    Decorator factory that creates an oracle constraint.

    Returns an :class:`OracleConstraint` when ``nodes`` is ``None``, and a
    :class:`~conin.constraints.FactorOracleConstraint` otherwise.

    Parameters
    ----------
    nodes : list or callable, optional
        Node scope for factor materialisation: a list of node names or a
        callable ``nodes(data)`` that yields node names.  See
        :class:`~conin.constraints.FactorOracleConstraint`.
    name : str, optional
        Human-readable constraint name.
    same_partial_as_func : bool, optional
        If ``True``, the partial-feasibility check reuses the main predicate.
        Not supported together with ``nodes``.

    Raises
    ------
    InvalidInputError
        If ``same_partial_as_func`` is set together with ``nodes``.
    """
    if nodes is not None and same_partial_as_func:
        raise InvalidInputError(
            "same_partial_as_func cannot be used with nodes: factor oracle "
            "constraints have no partial function."
        )

    def decorator(func):
        if nodes is not None:
            return FactorOracleConstraint(func=func, nodes=nodes, name=name)
        return OracleConstraint(
            func=func,
            name=name,
            same_partial_as_func=same_partial_as_func,
        )

    return decorator
