import numpy as np
import pytest

from conin import InvalidInputError
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models import HiddenMarkovModel
from conin.algorithms.sampling.unconstrained import (
    generate_hidden,
    generate_hidden_until_state,
    generate_observed,
    generate_observed_from_hidden,
)
from conin.util import Util

import conin.examples.hidden_markov_model as tc


class Test_HMM1:
    T = 25

    def test_generate_hidden_length(self):
        hmm = tc.create_hmm1()
        assert len(generate_hidden(hmm, self.T)) == self.T
        with pytest.raises(InvalidInputError):
            generate_hidden(hmm, -1)

    def test_generate_hidden_output(self):
        hmm = tc.create_hmm1()
        for h in generate_hidden(hmm, self.T):
            assert h in {"h0", "h1"}

    def test_generate_hidden_from_state(self):
        hmm = tc.create_hmm1()
        # Make sure that we start in state h0 so we test sequences of length at
        # least 2
        start_probs = hmm.get_start_probs()
        tranisition_probs = hmm.get_transition_probs()
        emission_probs = hmm.get_emission_probs()
        start_probs = {"h0": 1, "h1": 0}
        hmm = HiddenMarkovModel(
            start_probs=start_probs,
            emission_probs=emission_probs,
            transition_probs=tranisition_probs,
        )
        # This should be all h0's followed by h1
        vec = generate_hidden_until_state(hmm, "h1")
        assert all(val == "h0" for val in vec[:-1])

    def test_generate_observed_from_hidden_length(self):
        hmm = tc.create_hmm1()
        hidden = generate_hidden(hmm, self.T)
        assert len(generate_observed_from_hidden(hmm, hidden)) == self.T

    def test_generate_observed_from_hidden_output(self):
        hmm = tc.create_hmm1()
        hidden = generate_hidden(hmm, self.T)
        for o in generate_observed_from_hidden(hmm, hidden):
            assert o in {"o0", "o1"}

    def test_generate_observed_length(self):
        hmm = tc.create_hmm1()
        assert len(generate_observed(hmm, self.T)) == self.T
        with pytest.raises(InvalidInputError):
            generate_observed(hmm, -1)

    def test_generate_observed_output(self):
        hmm = tc.create_hmm1()
        for h in generate_observed(hmm, self.T):
            assert h in {"o0", "o1"}


def _reference_hidden(hmm, time_steps, rng):
    # The sampler the HMM had as a method: start, then one transition per step.
    start, transition = hmm.start_vec, hmm.transition_mat
    hidden = [Util.sample_from_vec(start, rng)]
    for t in range(time_steps - 1):
        hidden.append(Util.sample_from_vec(transition[hidden[t]], rng))
    return hidden


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_draws_as_the_reference_sampler(seed):
    hmm = tc.create_hmm2()
    rng = np.random.default_rng(seed)
    hidden = [hmm.hidden_to_external[h] for h in _reference_hidden(hmm, 15, rng)]
    emission = hmm.emission_mat
    observed = [
        hmm.observed_to_external[
            Util.sample_from_vec(emission[hmm.hidden_to_internal[h]], rng)
        ]
        for h in hidden
    ]
    rng = np.random.default_rng(seed)
    assert generate_hidden(hmm, 15, rng=rng) == hidden
    assert generate_observed_from_hidden(hmm, hidden, rng=rng) == observed
    assert generate_observed(hmm, 15, rng=seed) == observed


def test_unconstrained_constrained_model_gives_the_same_draws():
    hmm = tc.create_hmm2()
    expected = generate_observed(hmm, 10, rng=4)
    assert generate_observed(ConstrainedHiddenMarkovModel(hmm), 10, rng=4) == expected


@pytest.mark.parametrize(
    "sample",
    [
        lambda m, rng: generate_hidden(m, 8, rng=rng),
        lambda m, rng: generate_observed(m, 8, rng=rng),
        lambda m, rng: generate_observed_from_hidden(m, ["h0", "h1"] * 4, rng=rng),
        lambda m, rng: generate_hidden_until_state(m, "h2", rng=rng),
    ],
)
def test_rng_seed_and_generator(sample):
    hmm = tc.create_hmm2()
    # The same seed gives the same draws, a generator continues its stream, and
    # the global NumPy random state is neither used nor changed.
    np.random.seed(0)
    state = np.random.get_state()[1].copy()
    first = sample(hmm, 7)
    assert sample(hmm, 7) == first
    rng = np.random.default_rng(7)
    assert sample(hmm, rng) == first
    draws = [sample(hmm, rng) for _ in range(20)]
    assert any(d != first for d in draws)
    assert (np.random.get_state()[1] == state).all()


@pytest.mark.parametrize(
    "sample",
    [
        lambda m: generate_hidden(m, 5),
        lambda m: generate_observed(m, 5),
        lambda m: generate_observed_from_hidden(m, ["h0"] * 5),
        lambda m: generate_hidden_until_state(m, "h0"),
    ],
)
def test_constraints_are_not_ignored(sample):
    # Sampling a constrained model without its constraints would be silently
    # wrong; use rejection_sample_hidden for oracle constraints.
    with pytest.raises(InvalidInputError, match="matvec"):
        sample(tc.create_chmm1_oracle())


def test_until_state_gives_up_after_max_steps():
    # create_hmm2's h2 is absorbing, so h1 can become unreachable: this
    # looped forever.  The start state is h1 for 0.6 of the draws, so try
    # several seeds for one that reaches h2 first.
    hmm = tc.create_hmm2()
    gave_up = 0
    for seed in range(20):
        try:
            path = generate_hidden_until_state(hmm, "h1", max_steps=50, rng=seed)
        except RuntimeError as e:
            assert "within max_steps=50 states" in str(e)
            gave_up += 1
        else:
            assert path[-1] == "h1" and len(path) <= 50
    assert gave_up > 0


@pytest.mark.parametrize("max_steps,length", [(1, 1), (3, 3)])
def test_until_state_may_reach_the_target_at_max_steps(max_steps, length):
    # A path that reaches the target at exactly max_steps states is returned:
    # find a seed whose path to h0 has `length` states, then sample it again
    # with max_steps=length.
    hmm = tc.create_hmm2()
    for seed in range(200):
        try:
            path = generate_hidden_until_state(hmm, "h0", max_steps=1000, rng=seed)
        except RuntimeError:
            continue  # absorbed in h2 before reaching h0
        if len(path) == length:
            again = generate_hidden_until_state(
                hmm, "h0", max_steps=max_steps, rng=seed
            )
            assert again == path
            return
    pytest.fail(f"no seed gives a path of length {length}")


@pytest.mark.parametrize("max_steps", [0, -1])
def test_until_state_rejects_max_steps_below_one(max_steps):
    with pytest.raises(InvalidInputError, match="max_steps must be at least 1"):
        generate_hidden_until_state(tc.create_hmm1(), "h1", max_steps=max_steps)


def test_generate_observed_rejects_negative_time_steps_before_sampling():
    hmm = tc.create_hmm1()
    rng = np.random.default_rng(0)
    with pytest.raises(InvalidInputError):
        generate_observed(hmm, -1, rng=rng)
    assert rng.random() == np.random.default_rng(0).random()


@pytest.mark.parametrize("generate", [generate_hidden, generate_observed])
def test_zero_time_steps_gives_an_empty_sequence(generate):
    # It used to sample the start state anyway and return one label.
    hmm = tc.create_hmm1()
    rng = np.random.default_rng(0)
    assert generate(hmm, 0, rng=rng) == []
    assert rng.random() == np.random.default_rng(0).random()


@pytest.mark.parametrize("generate", [generate_hidden, generate_observed])
def test_negative_time_steps_message(generate):
    with pytest.raises(InvalidInputError, match="non-negative, got -2"):
        generate(tc.create_hmm1(), -2)
