"""Marginal and conditional log-probabilities on the ``"factor"`` representation.

The ``"factor"`` representation holds the model with its constraints compiled
in (:class:`~conin.representations.factor.FactorRepresentation`): a factor per
constraint that is ``0`` on infeasible assignments (Markov networks), or a
binary node per constraint that :attr:`constraint_evidence` fixes to ``1``
(Bayesian networks, and the unrolled DBNs and HMMs).  Let ``W(a)`` be the sum
of the product of the factors over every total assignment that agrees with
``a``, with the constraint evidence added.  Then ``W(a) = k * Z * P(a, C)``,
where ``Z`` is the partition function (``1`` for Bayesian networks), ``C`` is
the event that every constraint holds, and ``k`` is a constant of the compiled
constraint factors.  So

    log P(x | e, C) = log W(x, e) - log W(e)

and ``k`` and ``Z`` cancel.  ``W`` is computed with pgmpy's variable
elimination on the Markov network of the pgmpy model (pgmpy normalises the
queries of Bayesian networks, and drops factors that the evidence fixes
completely).  The sums are taken in linear space, so a probability below the
smallest float (about ``1e-308``, e.g. a very long HMM sequence) underflows
to zero.
"""

import math

from conin.exceptions import InvalidInputError
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.inference
    import pgmpy.models


def _log_probability_variable_elimination(repn, *, assignment, evidence):
    """``log P(assignment | evidence, C)`` on the ``"factor"`` representation ``repn``.

    ``assignment`` and ``evidence`` are dicts keyed by the nodes of
    ``repn.pgm`` (checked by the caller).

    Raises
    ------
    InvalidInputError
        If ``P(evidence, C) = 0``: the conditional is not defined.
    """
    model = repn.create_model()
    if isinstance(model, pgmpy.models.DiscreteBayesianNetwork):
        model = model.to_markov_model()
    infer = pgmpy.inference.VariableElimination(model)
    constraint_evidence = repn.constraint_evidence

    denominator = _weight(infer, model, {**evidence, **constraint_evidence})
    if denominator == 0:
        raise InvalidInputError(_zero_message(evidence, repn.constraints))
    numerator = _weight(infer, model, {**evidence, **assignment, **constraint_evidence})
    if numerator == 0:
        return -math.inf
    return math.log(numerator) - math.log(denominator)


def _weight(infer, model, fixed):
    """The sum of the product of the factors over the assignments that agree with ``fixed``."""
    if not fixed:
        node = next(iter(model.nodes))
        result = infer.query([node], show_progress=False)
        return float(result.values.sum())
    node = next(iter(fixed))
    rest = {n: s for n, s in fixed.items() if n != node}
    result = infer.query([node], evidence=rest, show_progress=False)
    return float(result.values[result.get_state_no(node, fixed[node])])


def _zero_message(evidence, constraints):
    if evidence and constraints:
        what = "the evidence and the constraints have"
    elif evidence:
        what = "the evidence has"
    else:
        what = "the constraints have"
    return f"The conditional probability is not defined: {what} probability zero"
