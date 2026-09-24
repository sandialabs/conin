import pytest

from conin.hidden_markov_model import learning
import math


def test_add_unknowns():
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]]) == [
        [0, 0, 1],
        ["__UNKNOWN__"],
        [0, 1, "__UNKNOWN__"],
        ["__UNKNOWN__"],
    ]
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]], num=2) == [
        [0, 0, "__UNKNOWN__"],
        ["__UNKNOWN__"],
        [0, "__UNKNOWN__", "__UNKNOWN__"],
        ["__UNKNOWN__"],
    ]
    assert learning.add_unknowns([[0, 0, 1], [2], [0, 1, 3], [4]], token="test") == [
        [0, 0, 1],
        ["test"],
        [0, 1, "test"],
        ["test"],
    ]


def test_add_unknowns_all_seen():
    # When every value appears more than num times, nothing is replaced.
    data = [["a", "b"], ["a", "b"]]
    result = learning.add_unknowns([["a", "b"], ["a", "b"]])
    assert result == [["a", "b"], ["a", "b"]]


def test_add_unknowns_empty_sublists():
    # Empty inner lists should not cause errors and should be passed through.
    result = learning.add_unknowns([[], []])
    assert result == [[], []]


def test_add_unknowns_mutates_input():
    # add_unknowns mutates and returns the same list object.
    data = [["x"], ["y"]]
    result = learning.add_unknowns(data)
    assert result is data


def test_convert_to_simulations():
    hidden = ["h0", "h1"]
    observed = ["o0", "o1"]
    sim = learning.convert_to_simulations(
        hidden_list=[hidden, hidden], observed_list=[observed, observed]
    )
    assert sim[0].hidden == hidden
    assert sim[0].observed == observed
    assert sim[0].index == 0
    assert sim[1].hidden == hidden
    assert sim[1].observed == observed
    assert sim[1].index == 1


def test_supervised_learning():
    eps = 0
    hidden_states = ["h0", "h1"]
    observable_states = ["o0", "o1"]

    hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
    observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=hidden_states,
        observable_states=observable_states,
        transition_tolerance=eps,
        emission_tolerance=eps,
        start_tolerance=eps,
    )

    start_probs = hmm.get_start_probs()
    emission_probs = hmm.get_emission_probs()
    transition_probs = hmm.get_transition_probs()

    assert math.isclose(start_probs["h0"], 0.5)
    assert math.isclose(start_probs["h1"], 0.5)
    assert math.isclose(transition_probs[("h0", "h0")], 0.5)
    assert math.isclose(transition_probs[("h0", "h1")], 0.5)
    assert math.isclose(transition_probs[("h1", "h0")], 1 / 3)
    assert math.isclose(transition_probs[("h1", "h1")], 2 / 3)
    assert math.isclose(emission_probs[("h0", "o0")], 2 / 3)
    assert math.isclose(emission_probs[("h0", "o1")], 1 / 3)
    assert math.isclose(emission_probs[("h1", "o0")], 1 / 4)
    assert math.isclose(emission_probs[("h1", "o1")], 3 / 4)


def test_supervised_learning_non_zero_tolerance():
    eps = 1e-4
    hidden_states = ["h0", "h1"]
    observable_states = ["o0", "o1"]

    hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
    observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=hidden_states,
        observable_states=observable_states,
        transition_tolerance=eps,
        emission_tolerance=eps,
        start_tolerance=eps,
    )

    start_probs = hmm.get_start_probs()
    emission_probs = hmm.get_emission_probs()
    transition_probs = hmm.get_transition_probs()

    assert math.isclose(start_probs["h0"], 0.5)
    assert math.isclose(start_probs["h1"], 0.5)
    assert math.isclose(transition_probs[("h0", "h0")], 0.5)
    assert math.isclose(transition_probs[("h0", "h1")], 0.5)
    assert math.isclose(transition_probs[("h1", "h0")], (1 + eps) / (3 + 2 * eps))
    assert math.isclose(transition_probs[("h1", "h1")], (2 + eps) / (3 + 2 * eps))
    assert math.isclose(emission_probs[("h0", "o0")], (2 + eps) / (3 + 2 * eps))
    assert math.isclose(emission_probs[("h0", "o1")], (1 + eps) / (3 + 2 * eps))
    assert math.isclose(emission_probs[("h1", "o0")], (1 + eps) / (4 + 2 * eps))
    assert math.isclose(emission_probs[("h1", "o1")], (3 + eps) / (4 + 2 * eps))


def test_supervised_learning_extra_hidden_observed():
    eps = 0
    hidden_states = ["h0", "h1", "h2"]
    observable_states = ["o0", "o1", "o2"]

    hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
    observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=hidden_states,
        observable_states=observable_states,
        transition_tolerance=eps,
        emission_tolerance=eps,
        start_tolerance=eps,
    )

    start_probs = hmm.get_start_probs()
    emission_probs = hmm.get_emission_probs()
    transition_probs = hmm.get_transition_probs()

    assert math.isclose(start_probs["h0"], 0.5)
    assert math.isclose(start_probs["h1"], 0.5)
    assert math.isclose(start_probs["h2"], 0)
    assert math.isclose(transition_probs[("h0", "h0")], 0.5)
    assert math.isclose(transition_probs[("h0", "h1")], 0.5)
    assert math.isclose(transition_probs[("h0", "h2")], 0)
    assert math.isclose(transition_probs[("h1", "h0")], 1 / 3)
    assert math.isclose(transition_probs[("h1", "h1")], 2 / 3)
    assert math.isclose(transition_probs[("h1", "h2")], 0)
    assert math.isclose(transition_probs[("h2", "h0")], 1 / 3)
    assert math.isclose(transition_probs[("h2", "h1")], 1 / 3)
    assert math.isclose(transition_probs[("h2", "h2")], 1 / 3)
    assert math.isclose(emission_probs[("h0", "o0")], 2 / 3)
    assert math.isclose(emission_probs[("h0", "o1")], 1 / 3)
    assert math.isclose(emission_probs[("h0", "o2")], 0)
    assert math.isclose(emission_probs[("h1", "o0")], 1 / 4)
    assert math.isclose(emission_probs[("h1", "o1")], 3 / 4)
    assert math.isclose(emission_probs[("h1", "o2")], 0)
    assert math.isclose(emission_probs[("h2", "o0")], 1 / 3)
    assert math.isclose(emission_probs[("h2", "o1")], 1 / 3)
    assert math.isclose(emission_probs[("h2", "o2")], 1 / 3)


def test_supervised_learning_priors():
    eps = 0
    hidden_states = ["h0", "h1", "h2"]
    observable_states = ["o0", "o1", "o2"]

    hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
    observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    transition_prior = {("h2", "h2"): 1}
    emission_prior = {("h2", "o2"): 1 / 2, ("h2", "o0"): 1 / 2}

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=hidden_states,
        observable_states=observable_states,
        transition_tolerance=eps,
        emission_tolerance=eps,
        start_tolerance=eps,
        transition_prior=transition_prior,
        emission_prior=emission_prior,
    )

    start_probs = hmm.get_start_probs()
    emission_probs = hmm.get_emission_probs()
    transition_probs = hmm.get_transition_probs()

    assert math.isclose(start_probs["h0"], 0.5)
    assert math.isclose(start_probs["h1"], 0.5)
    assert math.isclose(start_probs["h2"], 0)
    assert math.isclose(transition_probs[("h0", "h0")], 0.5)
    assert math.isclose(transition_probs[("h0", "h1")], 0.5)
    assert math.isclose(transition_probs[("h0", "h2")], 0)
    assert math.isclose(transition_probs[("h1", "h0")], 1 / 3)
    assert math.isclose(transition_probs[("h1", "h1")], 2 / 3)
    assert math.isclose(transition_probs[("h1", "h2")], 0)
    assert math.isclose(transition_probs[("h2", "h0")], 0)
    assert math.isclose(transition_probs[("h2", "h1")], 0)
    assert math.isclose(transition_probs[("h2", "h2")], 1)
    assert math.isclose(emission_probs[("h0", "o0")], 2 / 3)
    assert math.isclose(emission_probs[("h0", "o1")], 1 / 3)
    assert math.isclose(emission_probs[("h0", "o2")], 0)
    assert math.isclose(emission_probs[("h1", "o0")], 1 / 4)
    assert math.isclose(emission_probs[("h1", "o1")], 3 / 4)
    assert math.isclose(emission_probs[("h1", "o2")], 0)
    assert math.isclose(emission_probs[("h2", "o0")], 1 / 2)
    assert math.isclose(emission_probs[("h2", "o1")], 0)
    assert math.isclose(emission_probs[("h2", "o2")], 1 / 2)


def test_supervised_learning_no_hidden_states():
    sim = learning.convert_to_simulations(hidden_list=[["h0"]], observed_list=[["o0"]])
    with pytest.raises(AssertionError):
        learning.supervised_learning(
            simulations=sim,
            hidden_states=[],
            observable_states=["o0"],
        )


def test_supervised_learning_no_observable_states():
    sim = learning.convert_to_simulations(hidden_list=[["h0"]], observed_list=[["o0"]])
    with pytest.raises(AssertionError):
        learning.supervised_learning(
            simulations=sim,
            hidden_states=["h0"],
            observable_states=[],
        )


def test_supervised_learning_uniform_fallback_no_prior():
    # When a hidden state is declared but never appears in any simulation and
    # no prior is provided, transitions/emissions from that state should fall
    # back to uniform distributions.
    eps = 0
    hidden_states = ["h0", "h1", "h2"]
    observable_states = ["o0", "o1"]

    hidden = [["h0", "h1"]]
    observed = [["o0", "o1"]]

    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=hidden_states,
        observable_states=observable_states,
        transition_tolerance=eps,
        emission_tolerance=eps,
        start_tolerance=eps,
    )

    transition_probs = hmm.get_transition_probs()
    emission_probs = hmm.get_emission_probs()

    # h2 is never seen — should get uniform transition and emission fallbacks.
    assert math.isclose(transition_probs[("h2", "h0")], 1 / 3)
    assert math.isclose(transition_probs[("h2", "h1")], 1 / 3)
    assert math.isclose(transition_probs[("h2", "h2")], 1 / 3)
    assert math.isclose(emission_probs[("h2", "o0")], 1 / 2)
    assert math.isclose(emission_probs[("h2", "o1")], 1 / 2)


def test_supervised_learning_default_tolerances():
    # Calling without explicit tolerances should use default 1e-4 and
    # produce a valid HMM (probabilities sum to 1).
    hidden = [["h0", "h0", "h1"]]
    observed = [["o0", "o0", "o1"]]
    sim = learning.convert_to_simulations(hidden_list=hidden, observed_list=observed)

    hmm = learning.supervised_learning(
        simulations=sim,
        hidden_states=["h0", "h1"],
        observable_states=["o0", "o1"],
    )

    start_probs = hmm.get_start_probs()
    transition_probs = hmm.get_transition_probs()
    emission_probs = hmm.get_emission_probs()

    # Start probs must sum to 1.
    assert math.isclose(sum(start_probs.values()), 1.0)
    # Transition rows from each hidden state must sum to 1.
    for h in ["h0", "h1"]:
        row_sum = sum(transition_probs[(h, h2)] for h2 in ["h0", "h1"])
        assert math.isclose(row_sum, 1.0), f"Transition row for {h} does not sum to 1"
    # Emission rows from each hidden state must sum to 1.
    for h in ["h0", "h1"]:
        row_sum = sum(emission_probs[(h, o)] for o in ["o0", "o1"])
        assert math.isclose(row_sum, 1.0), f"Emission row for {h} does not sum to 1"


def test_mcem_runs_and_returns_log_prob():
    # Regression test for three bugs that were fixed in mcem.py:
    #   1. generate_hidden(observed) -> generate_hidden(len(observed))
    #   2. prev_log_prob != nan  -> not math.isnan(prev_log_prob)  (NaN != NaN is always True)
    #   3. math.abs(...)         -> abs(...)  (math.abs does not exist)
    #   4. app.hmm               -> app.hidden_markov_model  (HMM_MatVecRepn lacks log_probability)
    from conin.hidden_markov_model import learning
    import conin.hidden_markov_model.examples as tc

    chmm = tc.create_chmm1_oracle()
    chmm.hidden_markov_model.set_seed(0)
    observed = chmm.generate_observed(11)

    # mcem expects a ConstrainedHiddenMarkovModel as app, not the internal Oracle_CHMM.
    log_prob = learning.mcem(
        app=chmm,
        observed=observed,
        hidden_states=["h0", "h1"],
        observable_states=["o0", "o1"],
        max_iterations=3,
        samples_per_iteration=2,
    )
    assert isinstance(log_prob, float)
    assert log_prob < 0  # log-probability of any sequence is negative
