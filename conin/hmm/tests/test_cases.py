import conin.hmm

"""
This script has a collection of HMM test cases that can be used to
test conin capabilities.
"""

def create_hmm1():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    hmm = conin.hmm.HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_hmm2():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.8,
        ("h0", "h1"): 0.1,
        ("h0", "h2"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.7,
        ("h1", "h2"): 0.1,
        ("h2", "h2"): 1.0,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
        ("h2", "o2"): 1.0,
    }
    hmm = conin.hmm.HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    hmm.set_seed(0)
    return hmm


def create_chmm1():
    start_probs = {"h0": 0.4, "h1": 0.6}
    transition_probs = {
        ("h0", "h0"): 0.9,
        ("h0", "h1"): 0.1,
        ("h1", "h0"): 0.2,
        ("h1", "h1"): 0.8,
    }
    emission_probs = {
        ("h0", "o0"): 0.7,
        ("h0", "o1"): 0.3,
        ("h1", "o0"): 0.4,
        ("h1", "o1"): 0.6,
    }
    num_zeros_greater_than_nine = conin.hmm.Constraint(
        func=lambda seq: seq.count("h0") > 9,
        partial_func=lambda T, seq: T - seq.count("h0") >= 0,
    )
    num_zeros_less_than_thirteen = conin.hmm.Constraint(
        func=lambda seq: seq.count("h0") < 13,
        partial_func=lambda T, seq: T - seq.count("h0") < 13,
    )
    chmm = conin.hmm.Oracle_CHMM()
    chmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    chmm.add_constraint(num_zeros_greater_than_nine)
    chmm.add_constraint(num_zeros_less_than_thirteen)
    return chmm

