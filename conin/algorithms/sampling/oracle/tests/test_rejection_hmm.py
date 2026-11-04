"""Tests for conin.algorithms.sampling.oracle.rejection_sample_hidden."""

import importlib

import pytest

import conin.examples.hidden_markov_model as hex_
from conin.algorithms.feasibility import is_feasible
from conin.constrained_models import (
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import OracleConstraint, pyomo_constraint_fn
from conin.examples.markov_network import ABC_conin
from conin.exceptions import InvalidInputError
from conin.algorithms.sampling.oracle import rejection_sample_hidden
from conin.algorithms.sampling.unconstrained import generate_hidden

rejection_module = importlib.import_module(
    "conin.algorithms.sampling.oracle.rejection_hmm"
)


def _at_least_one(state):
    return OracleConstraint(
        func=lambda states: any(v == state for v in states.values()),
        name=f"at_least_one_{state}",
    )


def test_bare_hmm_draws_like_generate_hidden():
    hmm = hex_.create_hmm1()
    assert rejection_sample_hidden(hmm, 7, rng=3) == generate_hidden(hmm, 7, rng=3)


def test_unconstrained_cmodel_draws_like_generate_hidden():
    hmm = hex_.create_hmm1()
    expected = generate_hidden(hmm, 6, rng=5)
    assert (
        rejection_sample_hidden(ConstrainedHiddenMarkovModel(hmm), 6, rng=5) == expected
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_samples_are_feasible(seed):
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), _at_least_one("h0"))
    hidden = rejection_sample_hidden(chmm, 4, rng=seed)
    assert len(hidden) == 4
    assert set(hidden) <= {"h0", "h1"}
    assert "h0" in hidden
    assert is_feasible(chmm, hidden)


def test_example_oracle_constraints_are_satisfied():
    chmm = hex_.create_chmm1_oracle()
    hidden = rejection_sample_hidden(chmm, 15, rng=0)
    assert 9 < hidden.count("h0") < 13
    assert is_feasible(chmm, hidden)


def test_zero_time_steps_gives_an_empty_sequence():
    assert rejection_sample_hidden(hex_.create_hmm1(), 0) == []


def test_infeasible_constraints_raise_after_max_failures():
    impossible = OracleConstraint(func=lambda states: False, name="never")
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), impossible)
    with pytest.raises(RuntimeError, match="6 samples were rejected"):
        rejection_sample_hidden(chmm, 3, max_failures=5)


@pytest.mark.parametrize("max_failures", [0, 1, 5])
def test_every_drawn_sample_is_checked(monkeypatch, max_failures):
    # One more sample used to be drawn after the last rejection and never
    # checked (it still advanced the random state).
    drawn = []
    draw = rejection_module._hidden_indices

    def recording_draw(*args):
        drawn.append(draw(*args))
        return drawn[-1]

    monkeypatch.setattr(rejection_module, "_hidden_indices", recording_draw)
    checked = []

    def never(states):
        checked.append(list(states))
        return False

    chmm = ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), OracleConstraint(func=never, name="never")
    )
    with pytest.raises(RuntimeError):
        rejection_sample_hidden(chmm, 3, max_failures=max_failures)
    assert len(checked) == max_failures + 1
    assert len(drawn) == max_failures + 1


@pytest.mark.parametrize("hmm", [ABC_conin().pgm, "hmm", None])
def test_rejects_models_that_are_not_hmms(hmm):
    with pytest.raises(TypeError, match="expects a HiddenMarkovModel"):
        rejection_sample_hidden(hmm, 3)


def test_rejects_a_constrained_model_of_another_type():
    with pytest.raises(TypeError, match="expects a HiddenMarkovModel"):
        rejection_sample_hidden(ConstrainedDiscreteMarkovNetwork(ABC_conin().pgm), 3)


def test_rejects_constraints_without_an_oracle_representation():
    @pyomo_constraint_fn()
    def noop(M, D):
        return M

    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), noop)
    with pytest.raises(InvalidInputError, match="'oracle'"):
        rejection_sample_hidden(chmm, 3)


def test_rejects_negative_time_steps():
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), _at_least_one("h0"))
    with pytest.raises(InvalidInputError, match="non-negative"):
        rejection_sample_hidden(chmm, -1)
