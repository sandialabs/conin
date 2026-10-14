"""Materialisation of factor oracle constraints.

This module is a temporary home for materialisation until the factor
representation lands (restructure PR 3/4e).
"""

import itertools

from conin.constraints.factor_oracle import FactorOracleConstraint
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DiscreteCPD,
    DiscreteBayesianNetwork,
)


def materialise_factor_oracle(constraint, pgm, data=None):
    """Materialise a factor oracle constraint into a factor or CPD.

    All assignments over the constraint's node scope are enumerated and the
    predicate is evaluated on each of them.

    * For a :class:`DiscreteMarkovNetwork`, a :class:`DiscreteFactor` over the
      scope is returned whose value is ``10**6`` for feasible assignments and
      ``0`` otherwise.
    * For a :class:`DiscreteBayesianNetwork`, a :class:`DiscreteCPD` is
      returned for a new binary node named ``constraint.name`` whose parents
      are the scope; it takes the value ``1`` with probability 1 exactly when
      the parents' assignment is feasible.

    Parameters
    ----------
    constraint : FactorOracleConstraint
        The constraint to materialise.
    pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The probabilistic graphical model whose state spaces are used
        for enumeration.
    data : optional
        Application-specific data forwarded to the predicate when it
        accepts two arguments, and to ``constraint.nodes`` when it is
        callable.

    Returns
    -------
    DiscreteFactor or DiscreteCPD

    Raises
    ------
    TypeError
        If ``constraint`` is not a :class:`FactorOracleConstraint`.
    ValueError
        If ``pgm`` is neither a ``DiscreteMarkovNetwork`` nor a
        ``DiscreteBayesianNetwork``.
    """
    if not isinstance(constraint, FactorOracleConstraint):
        raise TypeError(
            f"Expected a FactorOracleConstraint, got {type(constraint).__name__}"
        )

    if type(constraint.nodes) is list:
        nodes = constraint.nodes
    else:
        nodes = list(constraint.nodes(data))

    if type(pgm) == DiscreteMarkovNetwork:
        M = 10**6
        values = {}
        for states in itertools.product(*(pgm.states[name] for name in nodes)):
            assignment = {name: states[i] for i, name in enumerate(nodes)}
            if constraint.num_args == 1:
                feasible = constraint.func(assignment)
            else:
                feasible = constraint.func(assignment, data)
            values[states] = M if feasible else 0
        return DiscreteFactor(nodes=nodes, values=values)

    elif type(pgm) == DiscreteBayesianNetwork:
        node = constraint.name
        values = {}
        for states in itertools.product(*(pgm.states[name] for name in nodes)):
            assignment = {name: states[i] for i, name in enumerate(nodes)}
            if constraint.num_args == 1:
                feasible = constraint.func(assignment)
            else:
                feasible = constraint.func(assignment, data)
            feasible = int(feasible)
            values[states] = {1: feasible, 0: 1 - feasible}
        return DiscreteCPD(node=node, parents=nodes, values=values)

    else:
        raise ValueError(f"Unexpected pgm: {type(pgm)}")
