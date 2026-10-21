"""Recursive constraints on HMM hidden-state sequences (used by recursive A*)."""

from ..exceptions import InvalidInputError
from .base import Constraint


def _always_true(state, t, T):
    return True


class RecursiveConstraint(Constraint):
    """A constraint on a hidden-state sequence, checked through a running state.

    The state summarises a prefix ``h_0, ..., h_{t-1}`` of the sequence: it is
    ``initial(h_0)`` for the first hidden state and ``update(state, h_t)`` for
    each later one.  Recursive A* (:func:`conin.inference.recursive.recursive_a_star`)
    keeps one search node per ``(last hidden state, prefix length, state)``,
    so prefixes with equal states are merged.

    Parameters
    ----------
    initial : callable
        ``initial(h) -> state``: the state after the first hidden state ``h``
        (an external hidden-state label).
    update : callable
        ``update(state, h) -> state``: the state after appending ``h``.
    feasible : callable
        ``feasible(state) -> bool``: whether a complete sequence with this
        state satisfies the constraint.
    feasible_partial : callable, optional
        ``feasible_partial(state, t, T) -> bool``: whether a prefix of length
        ``t`` with this state can still be completed to a feasible sequence of
        length ``T``.  It is a pruning test: it may return ``True`` for a
        prefix that cannot be completed, but must not return ``False`` for one
        that can.  Defaults to always ``True``.
    name : str, optional
        Human-readable name.  Defaults to ``feasible.__name__``.

    Notes
    -----
    States must be hashable.  When a model has several recursive constraints,
    the search state is the tuple of their states, and a sequence is feasible
    iff every constraint is satisfied.

    Examples
    --------
    The number of ``"h0"`` states is between ``lb`` and ``ub``:

    >>> from conin.constraints import RecursiveConstraint
    >>> lb, ub = 2, 3
    >>> num_h0 = RecursiveConstraint(
    ...     initial=lambda h: int(h == "h0"),
    ...     update=lambda n, h: n + int(h == "h0"),
    ...     feasible=lambda n: lb <= n <= ub,
    ...     feasible_partial=lambda n, t, T: n <= ub and n + (T - t) >= lb,
    ...     name="num_h0",
    ... )
    >>> num_h0.family
    'recursive'
    """

    family = "recursive"

    def __init__(self, *, initial, update, feasible, feasible_partial=None, name=None):
        for arg, value in (
            ("initial", initial),
            ("update", update),
            ("feasible", feasible),
        ):
            if not callable(value):
                raise InvalidInputError(
                    f"RecursiveConstraint: {arg} must be callable, got "
                    f"{type(value).__name__}"
                )
        if feasible_partial is not None and not callable(feasible_partial):
            raise InvalidInputError(
                "RecursiveConstraint: feasible_partial must be callable, got "
                f"{type(feasible_partial).__name__}"
            )
        self.initial = initial
        self.update = update
        self.feasible = feasible
        self.feasible_partial = (
            _always_true if feasible_partial is None else feasible_partial
        )
        self.name = name if name is not None else getattr(feasible, "__name__", None)
        if self.name is None:
            self.name = "Unnamed constraint"
