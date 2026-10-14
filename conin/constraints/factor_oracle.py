import inspect

from ..exceptions import InvalidInputError
from .base import Constraint


class FactorOracleConstraint(Constraint):
    """
    A predicate with an explicit node scope.

    A factor oracle constraint is materialised into a factor (Markov
    networks) or a CPD (Bayesian networks) by enumerating every assignment
    over its ``nodes`` (see
    :func:`conin.representations.factor.materialise_factor_oracle`).  This is
    how oracle constraints are used with variable elimination.

    Parameters
    ----------
    func : callable
        The constraint predicate ``func(assignment) -> bool``, or
        ``func(assignment, data) -> bool`` for constraints that need
        inference-time data.  ``assignment`` is a dict mapping node names to
        states.
    nodes : list or callable
        Node scope: a list of node names, or a callable ``nodes(data)`` that
        yields node names.
    name : str, optional
        Human-readable name.  Defaults to ``func.__name__``.

    Raises
    ------
    InvalidInputError
        If ``nodes`` is ``None``.
    """

    family = "factor_oracle"

    def __init__(self, *, func, nodes, name=None):
        if nodes is None:
            raise InvalidInputError(
                "FactorOracleConstraint requires a nodes scope (a list or a callable)."
            )
        self.func = func
        self.nodes = nodes
        self.num_args = len(inspect.signature(func).parameters)
        self.name = name if name is not None else func.__name__

    def __call__(self, assignment, data=None):
        """
        Evaluate the predicate on ``assignment``.

        Parameters
        ----------
        assignment : dict
            Mapping from node names to states.
        data : optional
            Inference-time data, passed to ``func`` when it takes two
            arguments.

        Returns
        -------
        bool
        """
        if self.num_args == 1:
            return self.func(assignment)
        return self.func(assignment, data)
