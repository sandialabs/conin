import copy

from conin.exceptions import InvalidInputError
from conin.graphical_models.markov_network import DiscreteMarkovNetwork
from conin.graphical_models.bayesian_network import (
    DiscreteCPD,
    DiscreteBayesianNetwork,
)
from conin.graphical_models.dynamic_bayesian_network import (
    DynamicDiscreteBayesianNetwork,
)
from conin.graphical_models.expr import ExpressionVariable


def create_mn_from_bn(bn):
    pgm = DiscreteMarkovNetwork(
        states=bn.states, factors=[cpd.to_factor() for cpd in bn.cpds]
    )
    pgm.check_model()

    return pgm


def _all_cpds(*args):
    for arg in args:
        if arg:
            for v in arg:
                yield v


def create_bn_from_dbn(*, dbn, start, stop):
    """Unroll a dynamic Bayesian network into a Bayesian network.

    Parameters
    ----------
    dbn : DynamicDiscreteBayesianNetwork
        The model to unroll. It is not changed.
    start : int
        The first time step. Only ``0`` is supported: the CPDs of the first
        unrolled time step are the DBN's initial CPDs, which describe time 0.
    stop : int
        The last time step (inclusive), ``stop >= start``. ``stop == start``
        unrolls one time step.

    Returns
    -------
    DiscreteBayesianNetwork
        The network over the nodes ``(name, t)`` for ``t`` in ``start..stop``
        (plus the DBN's static nodes).

    Raises
    ------
    InvalidInputError
        If ``start != 0`` or ``stop < start``.
    """
    if start != 0:
        # The transition CPDs at time `start` have parents at `start - 1`, which
        # are not unrolled, so the time-`start` nodes would keep the initial
        # CPDs of time 0 (this used to fail later with an AttributeError).
        # Starting later would need the marginal distribution at `start`.
        raise InvalidInputError(
            f"A dynamic Bayesian network can only be unrolled from start=0; "
            f"got start={start}"
        )
    if stop < start:
        raise InvalidInputError(
            f"A dynamic Bayesian network is unrolled over start..stop with "
            f"stop >= start; got start={start}, stop={stop}"
        )

    #
    # Copy non-dynamic states
    #
    states = copy.copy(dbn.states)

    #
    # Copy dynamic states to all time steps
    #
    for t in range(start, stop + 1):
        for v, s in dbn.dynamic_states.items():
            states[(v, t)] = s

    #
    # Copy cpds
    #
    # If the node or parents are dynamic tuples, then we treat the
    # cpd as a dynamic map and add it if the time steps for all dynamic variaables
    # are feasible.
    #
    # The loop below evaluates the CPD time expressions by setting the DBN's
    # shared time variable. Restore its value afterwards so that the input DBN
    # is left unchanged.
    t_value = dbn.t.value()
    try:
        cpds = _unroll_cpds(dbn, start, stop)
    finally:
        dbn.t.set_value(t_value)

    pgm = DiscreteBayesianNetwork(states=states, cpds=cpds)
    pgm.check_model()

    return pgm


def _unroll_cpds(dbn, start, stop):
    cpds = []
    for cpd in dbn.cpds:
        dynamic = False
        for v in _all_cpds([cpd.node], cpd.parents):
            if (
                type(v) is tuple
                and v[0] in dbn.dynamic_states
                and not (type(v[1]) is int)
            ):
                dynamic = True
                break

        if not dynamic:
            cpds.append(cpd)
            continue

        for t in range(start, stop + 1):
            dbn.t.set_value(t)

            if type(cpd.node) is tuple:
                curr = cpd.node[1].value()
                if curr < start:
                    continue
                node = (cpd.node[0], curr)
            else:
                node = cpd.node

            skip = False
            parents = []
            for v in _all_cpds(cpd.parents):
                if type(v) is tuple:
                    curr = v[1].value()
                    if curr < start:
                        skip = True
                        break
                    parents.append((v[0], curr))
                else:
                    parents.append(v)

            if skip:
                continue
            cpds.append(DiscreteCPD(node=node, parents=parents, values=cpd.values))

    return cpds


def create_dbn_from_hmm(hmm, debug=False):
    """Create a dynamic Bayesian network representation of an HMM.

    Parameters
    ----------
    hmm : HiddenMarkovModel
        Hidden Markov model to convert.
    debug : bool, optional
        If ``True``, print the generated CPDs.

    Returns
    -------
    DynamicDiscreteBayesianNetwork
        Dynamic Bayesian network with hidden-state and emission-state CPDs
        equivalent to the input HMM.
    """
    hidden_states = list(sorted(hmm.hidden_states))
    emission_states = list(sorted(hmm.observed_states))

    #
    # An HMM has two time-indexed random variables:
    #   H - hidden states
    #   E - emission states
    #
    # The CPDs below refer to the time index `t`. The same ExpressionVariable
    # instance is passed to the DBN constructor: `create_bn_from_dbn` later
    # calls `dbn.t.set_value(...)` and expects it to be the exact object
    # embedded in the CPDs' `node`/`parents` tuples.
    t = ExpressionVariable()

    start_probs = hmm.get_start_probs()
    transition_probs = hmm.get_transition_probs()
    emission_probs = hmm.get_emission_probs()

    # Specify the CPD for the initial hidden states (time 0)
    cpd_H_0 = DiscreteCPD(
        node=("H", 0),
        values=start_probs,
    )

    # Specify the CPD for the hidden states at time t
    cpd_H_t = DiscreteCPD(
        node=("H", t),
        parents=[("H", t - 1)],
        values={
            h0: {h1: transition_probs[h0, h1] for h1 in hidden_states}
            for h0 in hidden_states
        },
    )

    # Specify the CPD for the emission states at time t
    cpd_E_t = DiscreteCPD(
        node=("E", t),
        parents=[("H", t)],
        values={
            h: {e: emission_probs[h, e] for e in emission_states} for h in hidden_states
        },
    )

    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states={
            "H": hidden_states,
            "E": emission_states,
        },
        cpds=[cpd_H_0, cpd_H_t, cpd_E_t],
        t=t,
    )
    if debug:
        for cpd in dbn.cpds:
            print(cpd)

    return dbn
