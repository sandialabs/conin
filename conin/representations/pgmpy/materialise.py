"""Materialisation of factor oracle constraints.

Used by :func:`conin.representations.pgmpy.apply_pgmpy_constraints`, the
constraint applier of the ``"pgmpy"`` representations.
"""

import itertools

from conin.constraints.factor_oracle import FactorOracleConstraint
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DiscreteCPD,
    DiscreteBayesianNetwork,
)


def _key(states):
    """The key of an assignment in factor or CPD values: the state itself for one node.

    Values over one node are keyed by its states (``{0: w0, 1: w1}``), as in
    :class:`~conin.graphical_models.DiscreteFactor`, and over several nodes by
    tuples of states.
    """
    return states[0] if len(states) == 1 else states


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
            values[_key(states)] = M if feasible else 0
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
            values[_key(states)] = {1: feasible, 0: 1 - feasible}
        return DiscreteCPD(node=node, parents=nodes, values=values)

    else:
        raise ValueError(f"Unexpected pgm: {type(pgm)}")
