import functools
import inspect

from conin.constraints.base import Constraint, decorate
from conin.exceptions import InvalidInputError

from .factor import FactorOracleConstraint


def _always_feasible(T, states):
    return True


def _partial_from_func(func, T, states):
    return func(states)


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
        # Module-level functions rather than lambdas, so that the constraint
        # can be pickled.
        if same_partial_as_func is True:
            partial_func = functools.partial(_partial_from_func, func)
        elif partial_func is None:
            partial_func = _always_feasible

        # Determine how many positional args `func` accepts (1 or 2).
        num_args = len(inspect.signature(func).parameters) if func is not None else None

        # If no name is provided, use the function's name
        if name is None:
            name = func.__name__ if func is not None else "Unnamed constraint"
        self._set(func=func, partial_func=partial_func, num_args=num_args, name=name)

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


def oracle_accepts(constraint, assignment, data):
    """Evaluate an ``OracleConstraint`` on a total assignment.

    The predicate is called as ``func(assignment, data)`` if it takes two
    arguments, else as ``func(assignment)``.
    """
    if constraint.num_args == 2:
        return bool(constraint(assignment, data))
    return bool(constraint(assignment))


def oracle_constraint_fn(
    func=None, /, *, nodes=None, name=None, partial=None, same_partial_as_func=None
):
    """
    Decorator factory that creates an oracle constraint.

    Returns an :class:`OracleConstraint` when ``nodes`` is ``None``, and a
    :class:`~conin.constraints.FactorOracleConstraint` otherwise.  Use it as
    ``@oracle_constraint_fn`` or, with arguments,
    ``@oracle_constraint_fn(nodes=..., partial=...)``.

    Parameters
    ----------
    func : callable, optional
        The predicate, when the decorator is used without arguments.
    nodes : list or callable, optional
        Node scope for factor materialisation: a list of node names or a
        callable ``nodes(data)`` that yields node names.  See
        :class:`~conin.constraints.FactorOracleConstraint`.
    name : str, optional
        Human-readable constraint name.
    partial : callable, optional
        The partial-feasibility check ``partial(T, states) -> bool`` of an
        HMM oracle: ``states`` maps the time steps of a prefix to hidden
        states and ``T`` is the length of the full sequence.  It may accept a
        prefix that cannot be completed, but must not reject one that can;
        A* search skips the prefixes it rejects.  Not supported together with
        ``nodes`` or ``same_partial_as_func``.
    same_partial_as_func : bool, optional
        If ``True``, the partial-feasibility check reuses the main predicate.
        Not supported together with ``nodes``.

    Raises
    ------
    InvalidInputError
        If ``partial`` or ``same_partial_as_func`` is set together with
        ``nodes``, or both are set.
    """
    if nodes is not None and (same_partial_as_func or partial is not None):
        raise InvalidInputError(
            "partial and same_partial_as_func cannot be used with nodes: factor "
            "oracle constraints have no partial function."
        )
    if partial is not None and same_partial_as_func:
        raise InvalidInputError(
            "Give either partial or same_partial_as_func, not both."
        )

    def decorator(func):
        if nodes is not None:
            return FactorOracleConstraint(func=func, nodes=nodes, name=name)
        return OracleConstraint(
            func=func,
            name=name,
            partial_func=partial,
            same_partial_as_func=same_partial_as_func,
        )

    return decorate(func, decorator, "oracle_constraint_fn")
