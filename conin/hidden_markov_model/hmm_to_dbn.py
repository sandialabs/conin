from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
from conin.bayesian_network import DiscreteCPD


def create_dbn_from_hmm(hmm, debug=False):
    dbn = DynamicDiscreteBayesianNetwork()

    hidden_states = list(sorted(hmm.hidden_states))
    emission_states = list(sorted(hmm.observed_states))

    #
    # An HMM has two time-indexed random variables:
    #   H - hidden states
    #   E - emission states
    #
    dbn.dynamic_states = {
        "H": hidden_states,
        "E": emission_states,
    }

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

    # Add CPDs to the DDBN
    dbn.cpds = [cpd_H_0, cpd_H_t, cpd_E_t]
    if debug:
        for cpd in dbn.cpds:
            print(cpd)

    return dbn
