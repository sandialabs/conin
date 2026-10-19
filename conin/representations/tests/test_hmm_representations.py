"""Tests for the HMM representations ("oracle", "pyomo_network_flow", "mvr")."""

import munch
import pyomo.environ as pyo
import pytest

import conin.examples.hidden_markov_model as tc
from conin.config import default_mip_solver
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import (
    mvr_constraint_fn,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.constraints.mvr import BaseMVR
from conin.constraints.mvr.primitives import mvr_forbid_state, mvr_visit_state
from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.hidden_markov_model.chmm_oracle import Oracle_CHMM
from conin.inference.pyomo import ip_inference

# ---------------------------------------------------------------------
# "oracle"
# ---------------------------------------------------------------------


def test_oracle_matches_a_directly_built_oracle_chmm():
    cmodel = tc.create_chmm1_oracle()
    hmm = cmodel.model
    repn = cmodel.representation("oracle")
    direct = Oracle_CHMM(
        hmm=hmm.repn,
        constraints=list(cmodel.constraints),
        hidden_to_external=hmm.hidden_to_external,
    )

    for seq in ([0] * 9, [0] * 11, [0] * 15, [0] * 10 + [1] * 5, [1] * 12):
        assert repn.is_feasible(seq) == direct.is_feasible(seq)
    for T, seq in ((9, [0] * 9), (100, [0] * 9), (100, [0] * 15)):
        assert repn.partial_is_feasible(T=T, seq=seq) == direct.partial_is_feasible(
            T=T, seq=seq
        )

    hmm.set_seed(3)
    sampled = repn.generate_hidden(20)
    hmm.set_seed(3)
    assert sampled == direct.generate_hidden(20)


def test_oracle_without_constraints_accepts_everything():
    repn = ConstrainedHiddenMarkovModel(tc.create_hmm1()).representation("oracle")
    assert repn.constraints == []
    assert repn.is_feasible([0, 1, 0])
    assert repn.partial_is_feasible(T=10, seq=[0, 1])


def test_oracle_get_copy_returns_the_cached_object():
    cmodel = tc.create_chmm1_oracle()
    assert cmodel.representation("oracle") is cmodel.representation("oracle")


# ---------------------------------------------------------------------
# "pyomo_network_flow"
# ---------------------------------------------------------------------


def test_pyomo_network_flow_generates_the_constrained_model():
    repn = tc.create_chmm1_pyomo().representation("pyomo_network_flow")
    M, data = repn.create_model(observed=["o0"] * 15)
    assert isinstance(M.h0_lower, pyo.Constraint)
    assert isinstance(M.h0_upper, pyo.Constraint)
    assert data.hmm.Tmax == 15


def test_pyomo_network_flow_does_not_write_into_user_data():
    @pyomo_constraint_fn()
    def at_most_k_zeros(M, D):
        M.k_zeros = pyo.Constraint(expr=sum(M.V("H", t, "h0") for t in D.hmm.T) <= D.k)

    user_data = munch.Munch(k=1)
    cmodel = ConstrainedHiddenMarkovModel(
        tc.create_hmm1(), at_most_k_zeros, data=user_data
    )
    repn = cmodel.representation("pyomo_network_flow")
    M, data = repn.create_model(observed=["o0"] * 5)

    assert pyo.value(M.k_zeros.upper) == 1
    assert "hmm" in data
    assert "hmm" not in user_data
    assert user_data == munch.Munch(k=1)


def test_pyomo_network_flow_get_copy_returns_the_cached_object():
    cmodel = tc.create_chmm1_pyomo()
    first = cmodel.representation("pyomo_network_flow")
    assert cmodel.representation("pyomo_network_flow") is first
    # Each query builds a fresh model and fresh per-query data.
    M1, d1 = first.create_model(observed=["o0"] * 5)
    M2, d2 = first.create_model(observed=["o0"] * 7)
    assert M1 is not M2 and d1 is not d2
    assert d1.hmm.Tmax == 5 and d2.hmm.Tmax == 7


@pytest.mark.skipif(not default_mip_solver, reason="No mip solver installed")
def test_ip_inference_uses_the_pyomo_network_flow_representation():
    results = ip_inference(
        hmm=tc.create_chmm1_pyomo(),
        observed=["o0"] * 15,
        solver=default_mip_solver,
    )
    assert results.solution.hidden == ["h1"] * 3 + ["h0"] * 12


# ---------------------------------------------------------------------
# "mvr"
# ---------------------------------------------------------------------


def _hmm(hidden_states=("a", "b", "c")):
    hidden_states = list(hidden_states)
    n = len(hidden_states)
    observed_states = ["o1", "o2"]
    return HiddenMarkovModel(
        start_probs={h: 1.0 / n for h in hidden_states},
        transition_probs={
            (h1, h2): 1.0 / n for h1 in hidden_states for h2 in hidden_states
        },
        emission_probs={(h, o): 0.5 for h in hidden_states for o in observed_states},
    )


def test_mvr_factories_and_prebuilt_mvrs():
    hmm = _hmm()
    data = {"state": "b"}
    calls = []

    @mvr_constraint_fn(name="visit")
    def visit(hidden_markov_model, d):
        calls.append((hidden_markov_model, d))
        return mvr_visit_state(hidden_markov_model, [d["state"]])

    prebuilt = mvr_forbid_state(hmm, ["a"])
    cmodel = ConstrainedHiddenMarkovModel(hmm, [visit, prebuilt], data=data)
    repn = cmodel.representation("mvr")

    assert len(calls) == 1
    assert calls[0][0] is hmm and calls[0][1] is data
    assert isinstance(repn.mvrs, tuple)
    assert len(repn.mvrs) == 2
    assert all(isinstance(m, BaseMVR) for m in repn.mvrs)
    assert all(list(m.hidden_states) == hmm.hidden_to_external for m in repn.mvrs)
    # get_copy deep-copies, so repn.hmm is a copy of the model, not the model itself.
    assert repn.hmm.hidden_to_external == hmm.hidden_to_external


def test_mvr_mismatched_hidden_states_raise():
    hmm = _hmm()
    mvr = mvr_forbid_state(_hmm(("a", "b", "x")), ["a"])
    cmodel = ConstrainedHiddenMarkovModel(hmm, mvr)
    with pytest.raises(
        InvalidInputError, match="Hidden states of constraint 0 do not match"
    ):
        cmodel.representation("mvr")


def test_mvr_get_copy_is_a_deep_copy():
    hmm = _hmm()
    cmodel = ConstrainedHiddenMarkovModel(hmm, mvr_forbid_state(hmm, ["a"]))
    first = cmodel.representation("mvr")
    assert first.mvrs[0].time_range is None
    first.mvrs[0]._time_range = [0, 3]
    assert first.mvrs[0].time_range == [0, 3]

    second = cmodel.representation("mvr")
    assert second is not first
    assert second.mvrs[0].time_range is None


# ---------------------------------------------------------------------
# Unsupported (kind, family) pairs
# ---------------------------------------------------------------------


def test_unsupported_family_raises():
    @toulbar2_constraint_fn()
    def noop(M, D):
        pass

    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), noop)
    with pytest.raises(InvalidInputError) as excinfo:
        cmodel.representation("oracle")
    assert "('oracle', 'oracle')" in str(excinfo.value)
