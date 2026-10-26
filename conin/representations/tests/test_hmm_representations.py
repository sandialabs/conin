"""Tests for the HMM representations ("oracle", "recursive", "pyomo_network_flow", "mvr")."""

import copy
import pickle

import munch
import numpy as np
import pyomo.environ as pyo
import pytest

import conin.examples.hidden_markov_model as tc
from conin.config import default_mip_solver
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import (
    OracleConstraint,
    RecursiveConstraint,
    mvr_constraint_fn,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.constraints.mvr import BaseMVR, InhomMVR, MVR_MatVecRepn, mvr_timerange
from conin.constraints.mvr.primitives import mvr_forbid_state, mvr_visit_state
from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.mvr import MVRHMMRepresentation
from conin.representations.oracle import OracleHMMRepresentation
from conin.representations.recursive import RecursiveHMMRepresentation
from conin.sampling.oracle import rejection_sample_hidden
from conin.sampling.unconstrained import generate_hidden
from conin.inference.pyomo import ip_inference

# ---------------------------------------------------------------------
# "oracle"
# ---------------------------------------------------------------------


def _reference_is_feasible(cmodel, seq):
    """Reference check: every constraint on the external labels of ``seq``."""
    h2e = cmodel.model.hidden_to_external
    return all(c({t: h2e[h] for t, h in enumerate(seq)}) for c in cmodel.constraints)


def _reference_partial_is_feasible(cmodel, T, seq):
    h2e = cmodel.model.hidden_to_external
    states = {t: h2e[h] for t, h in enumerate(seq)}
    return all(c.partial_func(T, states) for c in cmodel.constraints)


def test_oracle_representation_class_and_attributes():
    cmodel = tc.create_chmm1_oracle()
    repn = cmodel.representation("oracle")
    assert isinstance(repn, OracleHMMRepresentation)
    assert repn.kind == "oracle"
    assert repn.model is cmodel.model
    assert isinstance(repn.hmm, MatVecHMMRepresentation)
    assert repn.hmm.model is cmodel.model
    # The user's constraints, not wrapped copies.
    assert repn.constraints is cmodel.constraints
    assert not hasattr(repn, "chmm")
    assert not hasattr(repn, "generate_hidden")


def test_oracle_matches_the_reference_checks():
    cmodel = tc.create_chmm1_oracle()
    repn = cmodel.representation("oracle")
    for seq in ([0] * 9, [0] * 11, [0] * 15, [0] * 10 + [1] * 5, [1] * 12):
        assert repn.is_feasible(seq) == _reference_is_feasible(cmodel, seq)
        assert repn.is_feasible(tuple(seq)) == _reference_is_feasible(cmodel, seq)
    for T, seq in ((9, [0] * 9), (100, [0] * 9), (100, [0] * 15), (12, [1] * 3)):
        assert repn.partial_is_feasible(T=T, seq=seq) == _reference_partial_is_feasible(
            cmodel, T, seq
        )


def test_oracle_agrees_with_is_feasible_on_external_labels():
    cmodel = tc.create_chmm1_oracle()
    repn = cmodel.representation("oracle")
    h2e = cmodel.model.hidden_to_external
    for seq in ([0] * 9, [0] * 11, [0] * 10 + [1] * 5, [1] * 12):
        assert repn.is_feasible(seq) == cmodel.is_feasible([h2e[h] for h in seq])


def test_oracle_two_argument_predicates_get_the_constraint_data():
    seen = []

    def at_most_half_h0(states, data):
        seen.append(data)
        return sum(1 for v in states.values() if v == "h0") <= len(data.hmm.T) // 2

    cmodel = ConstrainedHiddenMarkovModel(
        tc.create_hmm1(), OracleConstraint(func=at_most_half_h0)
    )
    repn = cmodel.representation("oracle")
    assert repn.is_feasible([0, 0, 1, 1])
    assert not repn.is_feasible([0, 0, 0, 1])
    assert seen[0] == munch.Munch(hmm=munch.Munch(T=[0, 1, 2, 3]))
    # The same answer as ConstrainedHiddenMarkovModel.is_feasible.
    assert cmodel.is_feasible(["h0", "h0", "h1", "h1"])
    assert not cmodel.is_feasible(["h0", "h0", "h0", "h1"])


def test_oracle_without_constraints_accepts_everything():
    repn = ConstrainedHiddenMarkovModel(tc.create_hmm1()).representation("oracle")
    assert repn.constraints == ()
    assert repn.is_feasible([0, 1, 0])
    assert repn.partial_is_feasible(T=10, seq=[0, 1])


def test_oracle_each_call_builds_a_new_representation():
    cmodel = tc.create_chmm1_oracle()
    assert cmodel.representation("oracle") is not cmodel.representation("oracle")


def test_oracle_checks_do_not_change_the_representation():
    cmodel = tc.create_chmm1_oracle()
    repn = cmodel.representation("oracle")
    before = dict(vars(repn))
    for seq in ([0] * 11, [1] * 12):
        repn.is_feasible(seq)
        repn.partial_is_feasible(T=20, seq=seq)
    assert vars(repn) == before


# ---------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------


def _reference_generate_hidden(cmodel, time_steps, max_failures=1000):
    """Reference rejection sampler: resample until every constraint holds."""
    hmm = cmodel.model
    h2i = hmm.hidden_to_internal
    hidden = [h2i[h] for h in generate_hidden(hmm, time_steps)]
    ctr = 0
    while not _reference_is_feasible(cmodel, hidden):
        hidden = [h2i[h] for h in generate_hidden(hmm, time_steps)]
        ctr += 1
        if ctr > max_failures:
            raise RuntimeError("failed")
    return [cmodel.model.hidden_to_external[h] for h in hidden]


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_rejection_sampling_matches_the_reference(seed):
    cmodel = tc.create_chmm1_oracle()
    cmodel.model.set_seed(seed)
    expected = [_reference_generate_hidden(cmodel, 20) for _ in range(3)]
    cmodel.model.set_seed(seed)
    new = [rejection_sample_hidden(cmodel, 20) for _ in range(3)]
    assert new == expected


def test_rejection_sampling_bare_hmm_is_unconstrained_sampling():
    hmm = tc.create_hmm1()
    hmm.set_seed(5)
    expected = generate_hidden(hmm, 12)
    hmm.set_seed(5)
    assert rejection_sample_hidden(hmm, 12) == expected
    hmm.set_seed(5)
    assert rejection_sample_hidden(ConstrainedHiddenMarkovModel(hmm), 12) == expected


def test_rejection_sampling_gives_up_after_max_failures():
    never = OracleConstraint(func=lambda states: False)
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), never)
    with pytest.raises(RuntimeError, match="after 3 trials"):
        rejection_sample_hidden(cmodel, 5, max_failures=3)


def test_rejection_sampling_rejects_other_families():
    with pytest.raises(InvalidInputError):
        rejection_sample_hidden(tc.create_chmm1_pyomo(), 5)
    with pytest.raises(InvalidInputError):
        rejection_sample_hidden(tc.create_chmm1_recursive(), 5)
    with pytest.raises(InvalidInputError):
        rejection_sample_hidden("not a model", 5)


# ---------------------------------------------------------------------
# "recursive"
# ---------------------------------------------------------------------


def _count(name, h0="h0"):
    return RecursiveConstraint(
        initial=lambda h: int(h == h0),
        update=lambda n, h: n + int(h == h0),
        feasible=lambda n: n >= 1,
        feasible_partial=lambda n, t, T: n + (T - t) >= 1,
        name=name,
    )


def test_recursive_representation_combines_states_into_tuples():
    c1 = _count("has_h0", "h0")
    c2 = _count("has_h1", "h1")
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), [c1, c2])
    repn = cmodel.representation("recursive")
    assert isinstance(repn, RecursiveHMMRepresentation)
    assert repn.kind == "recursive"
    assert repn.constraints is cmodel.constraints
    assert repn.model is cmodel.model

    s = repn.initial("h0")
    assert s == (1, 0)
    s = repn.update(s, "h0")
    assert s == (2, 0)
    assert not repn.feasible(s)
    assert repn.feasible_partial(s, 2, 3)
    assert not repn.feasible_partial(s, 2, 2)
    s = repn.update(s, "h1")
    assert s == (2, 1)
    assert repn.feasible(s)


def test_recursive_without_constraints():
    repn = ConstrainedHiddenMarkovModel(tc.create_hmm1()).representation("recursive")
    assert repn.initial("h0") == ()
    assert repn.update((), "h1") == ()
    assert repn.feasible(())
    assert repn.feasible_partial((), 1, 5)


def test_recursive_default_feasible_partial_is_always_true():
    c = RecursiveConstraint(
        initial=lambda h: h, update=lambda s, h: h, feasible=lambda s: s == "h0"
    )
    assert c.feasible_partial("h1", 1, 5)
    assert c.name == "<lambda>"
    assert c.family == "recursive"


def test_recursive_constraint_rejects_non_callables():
    with pytest.raises(InvalidInputError):
        RecursiveConstraint(initial=1, update=lambda s, h: s, feasible=lambda s: True)
    with pytest.raises(InvalidInputError):
        RecursiveConstraint(
            initial=lambda h: 0,
            update=lambda s, h: s,
            feasible=lambda s: True,
            feasible_partial="no",
        )


def test_recursive_each_call_builds_a_new_representation():
    cmodel = tc.create_chmm1_recursive()
    assert cmodel.representation("recursive") is not cmodel.representation("recursive")


def test_recursive_rejects_other_families():
    with pytest.raises(InvalidInputError):
        tc.create_chmm1_oracle().representation("recursive")
    with pytest.raises(InvalidInputError):
        tc.create_chmm1_recursive().representation("oracle")


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


def test_pyomo_network_flow_one_repn_serves_several_queries():
    cmodel = tc.create_chmm1_pyomo()
    first = cmodel.representation("pyomo_network_flow")
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
    assert repn.model is hmm


def test_mvr_mismatched_hidden_states_raise():
    hmm = _hmm()
    mvr = mvr_forbid_state(_hmm(("a", "b", "x")), ["a"])
    cmodel = ConstrainedHiddenMarkovModel(hmm, mvr)
    with pytest.raises(
        InvalidInputError, match="Hidden states of constraint 0 do not match"
    ):
        cmodel.representation("mvr")


def test_mvr_prebuilt_mvrs_are_not_shared_with_the_user():
    hmm = _hmm()
    prebuilt = mvr_forbid_state(hmm, ["a"])
    cmodel = ConstrainedHiddenMarkovModel(hmm, prebuilt)
    first = cmodel.representation("mvr")
    assert first.mvrs[0] is not prebuilt
    assert first.mvrs[0].time_range is None
    mvr_timerange(first.mvrs[0], [0, 3], inplace=True)
    assert first.mvrs[0].time_range == [0, 3]
    assert prebuilt.time_range is None

    second = cmodel.representation("mvr")
    assert second is not first
    assert second.mvrs[0].time_range is None


def test_mvr_representation_attributes():
    hmm = _hmm()
    repn = ConstrainedHiddenMarkovModel(
        hmm, mvr_forbid_state(hmm, ["a"])
    ).representation("mvr")
    assert isinstance(repn, MVRHMMRepresentation)
    assert repn.kind == "mvr"
    assert repn.model is hmm
    assert not hasattr(repn, "chmm")
    assert not hasattr(repn, "hmm")


@pytest.mark.parametrize("model", ["bare", "empty"])
def test_mvr_representation_without_constraints(model):
    hmm = _hmm()
    cmodel = ConstrainedHiddenMarkovModel(hmm) if model == "empty" else hmm
    repn = ConstrainedHiddenMarkovModel.ensure(cmodel).representation("mvr")
    assert repn.mvrs == ()
    assert repn.model is hmm


def test_mvr_representation_aligns_hidden_states_to_the_hmm():
    hmm = _hmm(("c", "a", "b"))  # internal order is sorted: a, b, c
    assert hmm.hidden_to_external == ["a", "b", "c"]
    mvr = mvr_forbid_state(hmm, ["a"])
    reordered = copy.copy(mvr)
    reordered.hidden_states = ["c", "a", "b"]
    perm = [2, 0, 1]  # reordered axis -> mvr axis
    reordered._repn = MVR_MatVecRepn(
        ini_array=mvr.repn.ini_array[perm],
        upd_array=mvr.repn.upd_array[perm],
        evl_array=mvr.repn.evl_array,
    )

    repn = ConstrainedHiddenMarkovModel(hmm, reordered).representation("mvr")
    (aligned,) = repn.mvrs
    assert aligned.hidden_states == ["a", "b", "c"]
    assert np.array_equal(aligned.repn.ini_array, mvr.repn.ini_array)
    assert np.array_equal(aligned.repn.upd_array, mvr.repn.upd_array)
    # the user's (deep-copied) MVR keeps its own order
    assert reordered.hidden_states == ["c", "a", "b"]


def test_mvr_representation_calls_factories_on_each_build():
    hmm = _hmm()
    calls = []

    @mvr_constraint_fn()
    def visit(hidden_markov_model):
        calls.append(hidden_markov_model)
        return mvr_visit_state(hidden_markov_model, ["b"])

    cmodel = ConstrainedHiddenMarkovModel(hmm, visit)
    first = cmodel.representation("mvr")
    second = cmodel.representation("mvr")
    assert len(calls) == 2
    assert first.mvrs[0] is not second.mvrs[0]
    # a new HMM (as in learning) gets its own MVRs
    other = _hmm()
    assert cmodel.with_model(other).representation("mvr").model is other
    assert calls[-1] is other


def test_mvr_arrays_are_read_only():
    hmm = _hmm()
    repn = ConstrainedHiddenMarkovModel(
        hmm, mvr_forbid_state(hmm, ["a"])
    ).representation("mvr")
    arrays = repn.mvrs[0].repn
    for name in ("ini_array", "upd_array", "evl_array", "ini_idx", "next_idx"):
        array = getattr(arrays, name)
        assert not array.flags.writeable, name
        with pytest.raises(ValueError):
            array[...] = 0
    for restored in (copy.deepcopy(arrays), pickle.loads(pickle.dumps(arrays))):
        assert not restored.ini_array.flags.writeable
        assert not restored.upd_array.flags.writeable


def test_mvr_inhom_arrays_are_read_only():
    hmm = _hmm(("a", "b"))
    mvr = InhomMVR(
        hidden_states=["a", "b"],
        mediation_states=[[0], [0]],
        ini={"a": 0, "b": 0},
        upd=[{(0, "a"): 0, (0, "b"): 0}],
        evl=[{0: True}, {0: True}],
    )
    arrays = ConstrainedHiddenMarkovModel(hmm, mvr).representation("mvr").mvrs[0].repn
    assert all(not a.flags.writeable for a in arrays.upd_array)
    assert all(not a.flags.writeable for a in arrays.evl_array)
    assert all(not a.flags.writeable for a in arrays.next_idx)


def test_mvr_matvec_repn_copies_its_inputs():
    ini = np.array([[1.0, 0.0], [0.0, 1.0]])
    upd = np.zeros((2, 2, 2))
    upd[:, 0, 0] = upd[:, 1, 1] = 1.0
    evl = np.array([1.0, 0.0])
    arrays = MVR_MatVecRepn(ini_array=ini, upd_array=upd, evl_array=evl)
    ini[0, 0] = 0.0  # the caller's array stays writeable and is not shared
    assert arrays.ini_array[0, 0] == 1.0


def test_mvr_representation_is_not_changed_by_mvr_timerange():
    hmm = _hmm()
    repn = ConstrainedHiddenMarkovModel(
        hmm, mvr_forbid_state(hmm, ["a"])
    ).representation("mvr")
    windowed = mvr_timerange(repn.mvrs[0], [1, 2])
    assert windowed.time_range == [1, 2]
    assert repn.mvrs[0].time_range is None


def test_mvr_other_families_raise():
    hmm = _hmm()
    cmodel = ConstrainedHiddenMarkovModel(hmm, OracleConstraint(func=lambda seq: True))
    with pytest.raises(InvalidInputError) as excinfo:
        cmodel.representation("mvr")
    assert "('mvr', 'mvr')" in str(excinfo.value)


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


# ---------------------------------------------------------------------
# The unrolled HMM representations need a stop time (it was an assert)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pyomo", "toulbar2", "pgmpy"])
def test_unrolled_hmm_representations_need_a_stop_time(kind):
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1())
    with pytest.raises(InvalidInputError, match="needs a stop time"):
        cmodel.representation(kind)


def test_network_flow_with_impossible_observations_raises():
    # No hidden state can emit "o2" (it was an assert).
    hmm = HiddenMarkovModel(
        start_probs={"h0": 1.0},
        transition_probs={("h0", "h0"): 1.0},
        emission_probs={("h0", "o0"): 1.0, ("h0", "o1"): 0.0},
    )
    repn = ConstrainedHiddenMarkovModel(hmm).representation("pyomo_network_flow")
    with pytest.raises(InvalidInputError, match="No feasible transitions"):
        repn.create_model(observed=["o0", "o1"])
