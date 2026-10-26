"""Tests for conin.sampling.oracle.rejection_sample_hidden."""

import numpy as np
import pytest

import conin.examples.hidden_markov_model as hex_
from conin.constrained_models import (
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import OracleConstraint, pyomo_constraint_fn
from conin.examples.markov_network import ABC_conin
from conin.exceptions import InvalidInputError
from conin.sampling.oracle import rejection_sample_hidden
from conin.sampling.unconstrained import generate_hidden


def _at_least_one(state):
    return OracleConstraint(
        func=lambda states: any(v == state for v in states.values()),
        name=f"at_least_one_{state}",
    )


def test_bare_hmm_draws_like_generate_hidden():
    hmm = hex_.create_hmm1()
    np.random.seed(3)
    expected = generate_hidden(hmm, 7)
    np.random.seed(3)
    assert rejection_sample_hidden(hmm, 7) == expected


def test_unconstrained_cmodel_draws_like_generate_hidden():
    hmm = hex_.create_hmm1()
    np.random.seed(5)
    expected = generate_hidden(hmm, 6)
    np.random.seed(5)
    assert rejection_sample_hidden(ConstrainedHiddenMarkovModel(hmm), 6) == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_samples_are_feasible(seed):
    np.random.seed(seed)
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), _at_least_one("h0"))
    hidden = rejection_sample_hidden(chmm, 4)
    assert len(hidden) == 4
    assert set(hidden) <= {"h0", "h1"}
    assert "h0" in hidden
    assert chmm.is_feasible(hidden)


def test_example_oracle_constraints_are_satisfied():
    np.random.seed(0)
    chmm = hex_.create_chmm1_oracle()
    hidden = rejection_sample_hidden(chmm, 15)
    assert 9 < hidden.count("h0") < 13
    assert chmm.is_feasible(hidden)


def test_zero_time_steps_gives_an_empty_sequence():
    assert rejection_sample_hidden(hex_.create_hmm1(), 0) == []


def test_infeasible_constraints_raise_after_max_failures():
    impossible = OracleConstraint(func=lambda states: False, name="never")
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1(), impossible)
    with pytest.raises(RuntimeError, match="after 5 trials"):
        rejection_sample_hidden(chmm, 3, max_failures=5)


@pytest.mark.parametrize("hmm", [ABC_conin().pgm, "hmm", None])
def test_rejects_models_that_are_not_hmms(hmm):
    with pytest.raises(InvalidInputError, match="expects a HiddenMarkovModel"):
        rejection_sample_hidden(hmm, 3)


def test_rejects_a_constrained_model_of_another_type():
    with pytest.raises(InvalidInputError, match="expects a HiddenMarkovModel"):
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
