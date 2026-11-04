"""Marginal and conditional log-probabilities on the ``"variable_elimination"`` representation.

The ``"variable_elimination"`` representation holds the model with its
constraints compiled in
(:class:`~conin.representations.variable_elimination.VariableEliminationRepresentation`):
the factors of the model (its CPDs for a Bayesian network) and one factor per
constraint that is ``0`` on infeasible assignments.  Let ``W(a)`` be the sum
of the product of these factors over every total assignment that agrees with
``a``.  Then ``W(a) = Z * P(a, C)``, where ``Z`` is the partition function
(``1`` for Bayesian networks) and ``C`` is the event that every constraint
holds.  So

    log P(x | e, C) = log W(x, e) - log W(e)

and ``Z`` cancels.  ``log W`` is computed with
:func:`~conin.algorithms.util.variable_elimination.log_weight`, in log space,
so long sequences do not underflow.
"""

import math

from conin.algorithms.util.variable_elimination import log_weight
from conin.exceptions import InvalidInputError


def _log_probability_variable_elimination(repn, *, assignment, evidence):
    """``log P(assignment | evidence, C)`` on the ``"variable_elimination"`` representation ``repn``.

    ``assignment`` and ``evidence`` are dicts keyed by the nodes of
    ``repn.pgm`` (checked by the caller).

    Raises
    ------
    InvalidInputError
        If ``P(evidence, C) = 0``: the conditional is not defined.
    """
    model = repn.ve_model
    denominator = log_weight(model.factors, model.states, evidence=evidence)
    if denominator == -math.inf:
        raise InvalidInputError(_zero_message(evidence, repn.constraints))
    numerator = log_weight(
        model.factors, model.states, evidence={**evidence, **assignment}
    )
    if numerator == -math.inf:
        return -math.inf
    return numerator - denominator


def _zero_message(evidence, constraints):
    if evidence and constraints:
        what = "the evidence and the constraints have"
    elif evidence:
        what = "the evidence has"
    else:
        what = "the constraints have"
    return f"The conditional probability is not defined: {what} probability zero"
