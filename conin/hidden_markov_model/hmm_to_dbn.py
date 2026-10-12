from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
from conin.bayesian_network import DiscreteCPD


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
    # We need `dbn.t` to exist before we can build the CPDs below (they
    # reference `dbn.t`/`dbn.t - 1`), so construct a throwaway dbn first with
    # just the dynamic states, grab its `.t`, and reuse that same
    # ExpressionVariable instance when constructing the final dbn below --
    # `create_bn_from_dbn` later calls `dbn.t.set_value(t)` and expects that
    # to be the exact object embedded in the CPDs' `node`/`parents` tuples.
    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states={
            "H": hidden_states,
            "E": emission_states,
        }
    )

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
        node=("H", dbn.t),
        parents=[("H", dbn.t - 1)],
        values={
            h0: {h1: transition_probs[h0, h1] for h1 in hidden_states}
            for h0 in hidden_states
        },
    )

    # Specify the CPD for the emission states at time t
    cpd_E_t = DiscreteCPD(
        node=("E", dbn.t),
        parents=[("H", dbn.t)],
        values={
            h: {e: emission_probs[h, e] for e in emission_states} for h in hidden_states
        },
    )

    # Build the final dbn, reusing the same dynamic_states and the same `.t`
    # ExpressionVariable instance embedded in the CPDs above.
    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states=dbn.dynamic_states,
        cpds=[cpd_H_0, cpd_H_t, cpd_E_t],
        t=dbn.t,
    )
    if debug:
        for cpd in dbn.cpds:
            print(cpd)

    return dbn
