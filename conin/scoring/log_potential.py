"""Log-probability and log-potential of a total assignment.

Both functions evaluate one assignment of every variable: no variable is summed
out, so no inference is needed.

* :func:`log_probability` returns ``log P(x)`` for the normalised models: Bayesian
  networks, dynamic Bayesian networks (unrolled over ``0..stop``) and HMMs.
* :func:`log_potential` returns the sum of the log factors (or CPD entries) of
  any model, including Markov networks, where it is ``log P(x) + log Z``.  For a
  constrained model it is ``-inf`` when a constraint is violated.

Marginal and conditional log-probabilities (with variables summed out, or given
evidence, or under the constraints) need inference: see
:func:`conin.scoring.log_probability_query`.

pgmpy models are converted with :func:`conin.common.pgmpy.convert_pgmpy_to_conin`.
"""

import math

from conin.common.pgmpy import convert_pgmpy_to_conin
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    GraphicalModel,
    HiddenMarkovModel,
    create_bn_from_dbn,
)
from conin.constrained_models import (
    ConstrainedGraphicalModel,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.feasibility import (
    check_total_assignment,
    dbn_horizon,
    hmm_sequences,
)


def log_probability(model, assignment, *, observed=None):
    """Log-probability of a total assignment of a Bayesian network, DBN or HMM.

    Parameters
    ----------
    model : DiscreteBayesianNetwork, DynamicDiscreteBayesianNetwork or HiddenMarkovModel
        A conin model, or a pgmpy ``DiscreteBayesianNetwork`` or
        ``DynamicBayesianNetwork`` (converted to conin).
    assignment : dict or list
        A state for every variable:

        * Bayesian network: ``{node: state}`` for every node;
        * DBN: ``{node: state}`` for every static node and ``{(name, t): state}``
          for every dynamic node and every ``t`` in ``0..stop``, where ``stop``
          is the largest time in the keys;
        * HMM: the hidden states, as a list or a dict keyed by ``0..T-1``
          (the format of ``map_query``'s HMM solutions).
    observed : list or dict, optional
        The observed states of an HMM (required for HMMs, and only allowed for
        them), with the same length as ``assignment``.

    Returns
    -------
    float
        ``log P(x)`` (for an HMM, ``log P(hidden, observed)``); ``-inf`` if the
        assignment has probability zero.

    Raises
    ------
    InvalidInputError
        For a Markov network (its probability needs the partition function; use
        :func:`log_potential`), a constrained model (its probability needs the
        probability of the constraints; use :func:`log_potential`), an
        assignment that is not total, or a missing or unexpected ``observed``.
    TypeError
        For an unsupported model type.

    Examples
    --------
    >>> from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
    >>> from conin.scoring import log_probability
    >>> bn = DiscreteBayesianNetwork(
    ...     states={"A": [0, 1], "B": [0, 1]},
    ...     cpds=[
    ...         DiscreteCPD(node="A", values=[0.9, 0.1]),
    ...         DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [0.9, 0.1]}),
    ...     ],
    ... )
    >>> round(log_probability(bn, {"A": 0, "B": 1}), 6)  # log(0.9 * 0.8)
    -0.328504
    """
    model = _as_conin(model)
    if isinstance(model, ConstrainedGraphicalModel):
        raise InvalidInputError(
            "log_probability is not defined for constrained models: the "
            "probability of an assignment under the constraints needs the "
            "probability of the constraints (an inference query). Use "
            "log_potential, which returns the unconstrained log-probability or "
            "-inf if a constraint is violated."
        )
    if isinstance(model, DiscreteMarkovNetwork):
        raise InvalidInputError(
            "log_probability is not defined for Markov networks: the "
            "probability needs the partition function (an inference query). "
            "Use log_potential for the unnormalised log-probability."
        )
    if isinstance(model, HiddenMarkovModel):
        if observed is None:
            raise InvalidInputError("log_probability of an HMM needs observed")
        return _log_hmm(model, assignment, observed)
    _no_observed(model, observed)
    if isinstance(model, DiscreteBayesianNetwork):
        return _log_bn(model, assignment)
    if isinstance(model, DynamicDiscreteBayesianNetwork):
        return _log_dbn(model, assignment)
    raise TypeError(f"Unexpected model type: {type(model)}")


def log_potential(model, assignment, *, observed=None):
    """Sum of the log factors of a total assignment, ``-inf`` if a constraint is violated.

    Parameters
    ----------
    model : GraphicalModel or ConstrainedGraphicalModel
        A conin Markov network, Bayesian network, DBN or HMM, a constrained
        one, or a pgmpy ``DiscreteMarkovNetwork``, ``DiscreteBayesianNetwork``
        or ``DynamicBayesianNetwork`` (converted to conin).
    assignment : dict or list
        A state for every variable, as for :func:`log_probability`; for a
        Markov network, ``{node: state}`` for every node.
    observed : list or dict, optional
        The observed states of an HMM (required for HMMs, and only allowed for
        them).

    Returns
    -------
    float
        * Markov network: ``sum log f(x)`` over the factors, which is
          ``log P(x) + log Z``.  A factor entry that is not given uses the
          factor's ``default_value``.
        * Bayesian network, DBN, HMM: the same as :func:`log_probability`.
        * Constrained model: the value for the unconstrained model, or ``-inf``
          if the assignment violates a constraint (see the models'
          ``is_feasible``).  It is not renormalised by the probability of the
          constraints.

    Raises
    ------
    InvalidInputError
        For an assignment that is not total, or a missing or unexpected
        ``observed``.
    NotImplementedError
        For a constrained model with ``"pyomo"`` or ``"toulbar2"`` constraints,
        whose functions can add variables to the backend model.
    TypeError
        For an unsupported model type.

    Examples
    --------
    >>> from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
    >>> from conin.scoring import log_potential
    >>> mn = DiscreteMarkovNetwork(
    ...     states={"A": [0, 1], "B": [0, 1]},
    ...     factors=[DiscreteFactor(nodes=["A", "B"], values=[1, 2, 3, 4])],
    ... )
    >>> round(log_potential(mn, {"A": 0, "B": 1}), 6)  # log(3): A varies fastest
    1.098612
    """
    model = _as_conin(model)
    if isinstance(model, ConstrainedGraphicalModel):
        # Check the assignment before the constraints see it.
        value = log_potential(model.model, assignment, observed=observed)
        if isinstance(model, ConstrainedHiddenMarkovModel):
            feasible = model.is_feasible(assignment, observed)
        else:
            feasible = model.is_feasible(assignment)
        return value if feasible else -math.inf
    if isinstance(model, DiscreteMarkovNetwork):
        _no_observed(model, observed)
        return _log_mn(model, assignment)
    return log_probability(model, assignment, observed=observed)


def _as_conin(model):
    if isinstance(model, (GraphicalModel, ConstrainedGraphicalModel)):
        return model
    if type(model).__module__.startswith("pgmpy."):
        return convert_pgmpy_to_conin(model)
    raise TypeError(f"Unexpected model type: {type(model)}")


def _no_observed(model, observed):
    if observed is not None:
        raise InvalidInputError(
            f"observed is only used for HMMs, not for {type(model).__name__}"
        )


def _log(x):
    if x == 0:
        return -math.inf
    return math.log(x)


def _log_mn(mn, assignment):
    check_total_assignment(mn.states, assignment)
    total = 0.0
    for factor in mn.factors:
        if len(factor.nodes) == 1:
            key = assignment[factor.nodes[0]]
        else:
            key = tuple(assignment[node] for node in factor.nodes)
        x = factor.values.get(key, factor.default_value)
        if x == 0:
            return -math.inf
        total += math.log(x)
    return total


def _log_bn(bn, assignment):
    check_total_assignment(bn.states, assignment)
    seen = set()
    for cpd in bn.cpds:
        if cpd.node in seen:
            # check_model() does not catch this; the product of the CPDs
            # would not be a probability.
            raise InvalidInputError(
                f"The Bayesian network has more than one CPD for node {cpd.node!r}"
            )
        seen.add(cpd.node)
    total = 0.0
    for cpd in bn.cpds:
        if cpd.parents:
            if len(cpd.parents) == 1:
                key = assignment[cpd.parents[0]]
            else:
                key = tuple(assignment[node] for node in cpd.parents)
            x = cpd.values.get(key, {}).get(assignment[cpd.node], cpd.default_value)
        else:
            x = cpd.values.get(assignment[cpd.node], cpd.default_value)
        if x == 0:
            return -math.inf
        total += math.log(x)
    return total


def _log_dbn(dbn, assignment):
    stop = dbn_horizon(dbn, assignment)
    if stop == 0:
        raise InvalidInputError(
            "A DBN assignment must cover at least the times 0..1 "
            "(create_bn_from_dbn needs start < stop)"
        )
    bn = create_bn_from_dbn(dbn=dbn, start=0, stop=stop)
    return _log_bn(bn, assignment)


def _log_hmm(hmm, hidden, observed):
    hidden, observed = hmm_sequences(hmm, hidden, observed)
    h = [hmm.hidden_to_internal[x] for x in hidden]
    o = [hmm.observed_to_internal[x] for x in observed]
    start, transition, emission = hmm.start_vec, hmm.transition_mat, hmm.emission_mat
    total = _log(start[h[0]]) + _log(emission[h[0]][o[0]])
    for t in range(1, len(h)):
        total += _log(transition[h[t - 1]][h[t]]) + _log(emission[h[t]][o[t]])
    return total
