# TODO we need way more here
import pytest

from conin.hmm import learning
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
