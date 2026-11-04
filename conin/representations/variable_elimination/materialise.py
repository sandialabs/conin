"""Materialisation of factor oracle constraints.

Used by :func:`~conin.representations.variable_elimination.apply_variable_elimination_constraints`,
the constraint applier of the ``"variable_elimination"`` representations.
"""

import itertools

from conin.constraints.oracle import FactorOracleConstraint
from conin.graphical_models import DiscreteFactor


def materialise_factor_oracle(constraint, pgm, data=None):
    """Materialise a factor oracle constraint into an indicator factor.

    All assignments over the constraint's node scope are enumerated and the
    predicate is evaluated on each of them.  The factor over the scope is
    ``1`` on feasible assignments and ``0`` otherwise, so multiplying it into
    a model removes the infeasible assignments.

    Parameters
    ----------
    constraint : FactorOracleConstraint
        The constraint to materialise.
    pgm : GraphicalModel
        The model whose state spaces are enumerated (only ``pgm.states`` is
        used).
    data : optional
        Application-specific data forwarded to the predicate when it
        accepts two arguments, and to ``constraint.nodes`` when it is
        callable.

    Returns
    -------
    DiscreteFactor
        The values are keyed by the state of the node for a scope of one node,
        and by tuples of states otherwise.

    Raises
    ------
    TypeError
        If ``constraint`` is not a :class:`FactorOracleConstraint`.
    """
    if not isinstance(constraint, FactorOracleConstraint):
        raise TypeError(
            f"Expected a FactorOracleConstraint, got {type(constraint).__name__}"
        )

    if not callable(constraint.nodes):
        nodes = constraint.nodes
    else:
        nodes = list(constraint.nodes(data))

    values = {}
    for states in itertools.product(*(pgm.states[name] for name in nodes)):
        assignment = dict(zip(nodes, states))
        if constraint.num_args == 1:
            feasible = constraint.func(assignment)
        else:
            feasible = constraint.func(assignment, data)
        values[states[0] if len(nodes) == 1 else states] = 1 if feasible else 0
    return DiscreteFactor(nodes=nodes, values=values)
