from conin.graphical_models.markov_network import DiscreteMarkovNetwork
from conin.graphical_models.bayesian_network import DiscreteCPD
from conin.graphical_models.dynamic_bayesian_network import (
    DynamicDiscreteBayesianNetwork,
)
from conin.graphical_models.time_index import TimeIndex


def create_mn_from_bn(bn):
    pgm = DiscreteMarkovNetwork(
        states=bn.states, factors=[cpd.to_factor() for cpd in bn.cpds]
    )
    pgm.check_model()

    return pgm


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
    emission_states = list(sorted(hmm.observable_states))

    #
    # An HMM has two time-indexed random variables:
    #   H - hidden states
    #   E - emission states
    #
    # The CPDs below refer to the time index `t`. The same TimeIndex
    # instance is passed to the DBN constructor: `create_bn_from_dbn` later
    # calls `dbn.t.set_value(...)` and expects it to be the exact object
    # embedded in the CPDs' `node`/`parents` tuples.
    t = TimeIndex()

    start_probs = hmm.start_probs
    transition_probs = hmm.transition_probs
    emission_probs = hmm.emission_probs

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
