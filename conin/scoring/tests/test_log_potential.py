"""Tests of conin.scoring.log_probability and conin.scoring.log_potential."""

import math

import pytest

from conin.scoring import log_potential, log_probability
from conin.constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
)
from conin.constraints import (
    FactorOracleConstraint,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    ExpressionVariable,
    HiddenMarkovModel,
)
from conin.util import try_import

import conin.examples.bayesian_network as bn_examples
import conin.examples.dynamic_bayesian_network as dbn_examples
import conin.examples.hidden_markov_model as hmm_examples
import conin.examples.markov_network as mn_examples

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as smoek_available:
    import smoek  # noqa: F401

require_pgmpy = pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
require_smoek = pytest.mark.skipif(not smoek_available, reason="smoek not installed")


def _bn():
    return DiscreteBayesianNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        cpds=[
            DiscreteCPD(node="A", values=[0.9, 0.1]),
            DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [1.0, 0.0]}),
        ],
    )


def _dbn():
    t = ExpressionVariable()
    return DynamicDiscreteBayesianNetwork(
        dynamic_states={"A": [0, 1], "B": [0, 1]},
        cpds=[
            DiscreteCPD(node=("A", 0), values=[0.9, 0.1]),
            DiscreteCPD(
                node=("B", t), parents=[("A", t)], values={0: [0.2, 0.8], 1: [0.9, 0.1]}
            ),
            DiscreteCPD(
                node=("A", t),
                parents=[("A", t - 1)],
                values={0: [0.2, 0.8], 1: [0.9, 0.1]},
            ),
        ],
        t=t,
    )


def _dbn_by_hand(a):
    """log P of an assignment of _dbn() over 0..stop."""
    A = {0: [0.2, 0.8], 1: [0.9, 0.1]}
    B = {0: [0.2, 0.8], 1: [0.9, 0.1]}
    stop = max(t for _, t in a)
    p = [0.9, 0.1][a["A", 0]]
    for t in range(stop + 1):
        p *= B[a["A", t]][a["B", t]]
        if t > 0:
            p *= A[a["A", t - 1]][a["A", t]]
    return math.log(p)


def _hmm():
    return HiddenMarkovModel(
        start_probs={"h0": 0.6, "h1": 0.4},
        transition_probs={
            ("h0", "h0"): 0.7,
            ("h0", "h1"): 0.3,
            ("h1", "h0"): 0.0,
            ("h1", "h1"): 1.0,
        },
        emission_probs={
            ("h0", "o0"): 0.9,
            ("h0", "o1"): 0.1,
            ("h1", "o0"): 0.2,
            ("h1", "o1"): 0.8,
        },
    )


# ---------------------------------------------------------------------
# Markov networks
# ---------------------------------------------------------------------


def test_mn_log_potential():
    pgm = mn_examples.ABC_conin().pgm
    assert log_potential(pgm, {"A": 2, "B": 2, "C": 1}) == pytest.approx(
        2.4849066497880004
    )


def test_mn_string_states_and_list_values():
    # List values are read with the first node varying fastest.
    mn = DiscreteMarkovNetwork(
        states={"A": ["x", "y"], "B": ["u", "v"]},
        factors=[DiscreteFactor(nodes=["A", "B"], values=[1, 2, 3, 4])],
    )
    assert log_potential(mn, {"A": "x", "B": "u"}) == pytest.approx(math.log(1))
    assert log_potential(mn, {"A": "y", "B": "u"}) == pytest.approx(math.log(2))
    assert log_potential(mn, {"A": "x", "B": "v"}) == pytest.approx(math.log(3))
    assert log_potential(mn, {"A": "y", "B": "v"}) == pytest.approx(math.log(4))


def test_mn_missing_factor_entries_use_default_value():
    """A sparse dict factor used to raise KeyError for a missing entry."""
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        factors=[
            DiscreteFactor(nodes=["A", "B"], values={(0, 0): 2.0}, default_value=1.0),
            DiscreteFactor(nodes=["A"], values={0: 3.0}),
        ],
    )
    assert log_potential(mn, {"A": 0, "B": 0}) == pytest.approx(math.log(6))
    assert log_potential(mn, {"A": 0, "B": 1}) == pytest.approx(math.log(3))
    # The second factor's default is 0.
    assert log_potential(mn, {"A": 1, "B": 1}) == -math.inf


def test_mn_zero_factor_is_minus_inf():
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1]}, factors=[DiscreteFactor(nodes=["A"], values=[0, 2])]
    )
    assert log_potential(mn, {"A": 0}) == -math.inf
    assert log_potential(mn, {"A": 1}) == pytest.approx(math.log(2))


def test_mn_log_probability_is_refused():
    with pytest.raises(InvalidInputError, match="partition function"):
        log_probability(mn_examples.ABC_conin().pgm, {"A": 2, "B": 2, "C": 1})


# ---------------------------------------------------------------------
# Bayesian networks
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [log_probability, log_potential])
def test_bn(f):
    bn = _bn()
    assert f(bn, {"A": 0, "B": 1}) == pytest.approx(math.log(0.9 * 0.8))
    assert f(bn, {"A": 1, "B": 0}) == pytest.approx(math.log(0.1))
    assert f(bn, {"A": 1, "B": 1}) == -math.inf


def test_bn_probabilities_sum_to_one():
    bn = bn_examples.cancer1_BN_conin().pgm
    import itertools

    nodes = bn.nodes
    total = sum(
        math.exp(log_probability(bn, dict(zip(nodes, values))))
        for values in itertools.product(*(bn.states_of(n) for n in nodes))
    )
    assert total == pytest.approx(1.0)


def test_bn_string_states_and_flat_list_values():
    bn = DiscreteBayesianNetwork(
        states={"A": ["a0", "a1"], "B": ["b0", "b1"]},
        cpds=[
            DiscreteCPD(node="A", values=[0.3, 0.7]),
            DiscreteCPD(node="B", parents=["A"], values=[0.9, 0.1, 0.2, 0.8]),
        ],
    )
    assert log_probability(bn, {"A": "a1", "B": "b1"}) == pytest.approx(
        math.log(0.7 * 0.8)
    )


def test_bn_two_parents():
    bn = bn_examples.cancer1_BN_conin().pgm
    a = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
    # P(Pollution=0) P(Smoker=1) P(Cancer=1|Smoker=1,Pollution=0)
    # P(Dyspnoea=0|Cancer=1) P(Xray=1|Cancer=1)
    expected = math.log(0.9 * 0.7 * 0.999 * 0.3 * 0.8)
    assert log_probability(bn, a) == pytest.approx(expected)


# ---------------------------------------------------------------------
# Dynamic Bayesian networks
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [log_probability, log_potential])
@pytest.mark.parametrize("stop", [1, 3])
def test_dbn(f, stop):
    import itertools

    dbn = _dbn()
    nodes = [(n, t) for t in range(stop + 1) for n in "AB"]
    total = 0.0
    for values in itertools.product([0, 1], repeat=len(nodes)):
        a = dict(zip(nodes, values))
        lp = f(dbn, a)
        assert lp == pytest.approx(_dbn_by_hand(a))
        total += math.exp(lp)
    assert total == pytest.approx(1.0)


def test_dbn_example():
    dbn = dbn_examples.simple1_DDBN_conin().pgm
    a = {("A", 0): 0, ("A", 1): 0, ("B", 0): 1, ("B", 1): 1}
    assert log_probability(dbn, a) == pytest.approx(math.log(0.9 * 0.8 * 0.2 * 0.8))


def test_dbn_does_not_change_t():
    dbn = _dbn()
    dbn.t.set_value(7)
    log_probability(dbn, {("A", 0): 0, ("A", 1): 0, ("B", 0): 1, ("B", 1): 1})
    assert dbn.t.value() == 7


@pytest.mark.parametrize(
    "assignment, match",
    [
        ({("A", 0): 0, ("B", 0): 1}, "0..1"),
        ({("A", 1): 0, ("B", 1): 1, ("A", 2): 0, ("B", 2): 1}, "0..stop"),
        ({("A", 0): 0, ("B", 0): 1, ("A", 2): 0, ("B", 2): 1}, "0..stop"),
        ({("A", 0): 0, ("A", 1): 0, ("B", 0): 1}, "missing"),
        ({"A": 0}, "no dynamic nodes"),
        ([0, 1], "must be a dict"),
    ],
)
def test_dbn_bad_assignment(assignment, match):
    with pytest.raises(InvalidInputError, match=match):
        log_probability(_dbn(), assignment)


# ---------------------------------------------------------------------
# HMMs
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [log_probability, log_potential])
def test_hmm(f):
    hmm = _hmm()
    expected = math.log(0.6 * 0.9 * 0.3 * 0.8 * 1.0 * 0.2)
    assert f(hmm, ["h0", "h1", "h1"], observed=["o0", "o1", "o0"]) == pytest.approx(
        expected
    )
    # dicts keyed by time, as in map_query's HMM solutions
    assert f(
        hmm, {0: "h0", 1: "h1", 2: "h1"}, observed={0: "o0", 1: "o1", 2: "o0"}
    ) == pytest.approx(expected)


def test_hmm_matches_method():
    hmm = hmm_examples.create_hmm1()
    hidden, observed = ["h1", "h0", "h0"], ["o0", "o0", "o1"]
    assert log_probability(hmm, hidden, observed=observed) == pytest.approx(
        hmm.log_probability(observed, hidden)
    )


def test_hmm_zero_probability_is_minus_inf():
    # HiddenMarkovModel.log_probability raises ValueError (math.log(0)) here.
    assert log_probability(_hmm(), ["h1", "h0"], observed=["o0", "o0"]) == -math.inf


@pytest.mark.parametrize(
    "hidden, observed, match",
    [
        (["h0"], None, "needs observed"),
        (["h0", "h0"], ["o0"], "same length"),
        (["hx"], ["o0"], "not a hidden state"),
        (["h0"], ["ox"], "not an observed state"),
        ([], [], "must not be empty"),
        ({1: "h0"}, ["o0"], "0..T-1"),
        ("h0", ["o0"], "list, a tuple or a dict"),
    ],
)
def test_hmm_bad_input(hidden, observed, match):
    with pytest.raises(InvalidInputError, match=match):
        log_probability(_hmm(), hidden, observed=observed)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [log_probability, log_potential])
@pytest.mark.parametrize(
    "assignment, match",
    [
        ({"A": 0}, "missing"),
        ({"A": 0, "B": 1, "C": 0}, "unknown nodes"),
        ({"A": 0, "B": 2}, "not a state of node 'B'"),
        ([0, 1], "must be a dict"),
    ],
)
def test_not_a_total_assignment(f, assignment, match):
    with pytest.raises(InvalidInputError, match=match):
        f(_bn(), assignment)


@pytest.mark.parametrize("f", [log_probability, log_potential])
def test_observed_only_for_hmms(f):
    with pytest.raises(InvalidInputError, match="only used for HMMs"):
        f(_bn(), {"A": 0, "B": 1}, observed=[0])


def test_no_evidence_argument():
    """``evidence`` (which returned the joint, not the conditional) is removed."""
    with pytest.raises(TypeError):
        log_potential(_bn(), {"A": 0}, evidence={"B": 1})


@pytest.mark.parametrize("f", [log_probability, log_potential])
def test_unsupported_model(f):
    with pytest.raises(TypeError, match="Unexpected model type"):
        f(object(), {})


# ---------------------------------------------------------------------
# Constrained models
# ---------------------------------------------------------------------


def _never(states):
    return False


def test_constrained_log_probability_is_refused():
    cmodel = ConstrainedDiscreteBayesianNetwork(_bn())
    with pytest.raises(InvalidInputError, match="constrained models"):
        log_probability(cmodel, {"A": 0, "B": 1})


def test_constrained_without_constraints():
    cmodel = ConstrainedDiscreteBayesianNetwork(_bn())
    assert cmodel.is_feasible({"A": 0, "B": 1})
    assert log_potential(cmodel, {"A": 0, "B": 1}) == pytest.approx(math.log(0.9 * 0.8))


def test_constrained_mn_factor_oracle():
    mn = mn_examples.ABC_conin().pgm
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mn, FactorOracleConstraint(func=lambda s: s["A"] != s["B"], nodes=["A", "B"])
    )
    assert log_potential(cmodel, {"A": 2, "B": 1, "C": 1}) == pytest.approx(
        log_potential(mn, {"A": 2, "B": 1, "C": 1})
    )
    assert log_potential(cmodel, {"A": 2, "B": 2, "C": 1}) == -math.inf


def test_constrained_bn_factor_oracle():
    example = bn_examples.cancer1_BN_constrained_oracle_conin()
    cmodel = example.pgm
    a = dict(example.solutions[0].states)
    assert cmodel.is_feasible(a)
    assert log_potential(cmodel, a) == pytest.approx(log_probability(cmodel.model, a))
    a["Xray"] = a["Dyspnoea"]
    assert not cmodel.is_feasible(a)
    assert log_potential(cmodel, a) == -math.inf


def test_constrained_dbn_factor_oracle():
    cmodel = dbn_examples.simple1_DDBN_constrained_oracle_conin().pgm
    a = {("A", 0): 0, ("A", 1): 0, ("B", 0): 1, ("B", 1): 1}
    assert log_potential(cmodel, a) == pytest.approx(log_probability(cmodel.model, a))
    a["A", 1] = 1
    assert log_potential(cmodel, a) == -math.inf


def test_constrained_dbn_factor_oracle_sees_data_T():
    seen = []

    def func(states, data):
        seen.append(data.T)
        return True

    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(
        _dbn(), FactorOracleConstraint(func=func, nodes=[("A", 0)])
    )
    a = {(n, t): 0 for n in "AB" for t in range(3)}
    assert cmodel.is_feasible(a)
    assert seen == [[0, 1, 2]]


def test_constrained_hmm_oracle():
    cmodel = hmm_examples.create_chmm1_oracle()
    hmm = cmodel.model
    observed = ["o0"] * 11
    assert log_potential(cmodel, ["h0"] * 11, observed=observed) == pytest.approx(
        log_probability(hmm, ["h0"] * 11, observed=observed)
    )
    assert (
        log_potential(cmodel, ["h0"] * 9 + ["h1"] * 2, observed=observed) == -math.inf
    )


def test_constrained_hmm_needs_observed():
    with pytest.raises(InvalidInputError, match="needs observed"):
        log_potential(hmm_examples.create_chmm1_oracle(), ["h0"] * 11)


def test_constrained_checks_assignment_before_constraints():
    called = []

    def func(states):
        called.append(states)
        return True

    cmodel = ConstrainedDiscreteBayesianNetwork(
        _bn(), FactorOracleConstraint(func=func, nodes=["A", "B"])
    )
    with pytest.raises(InvalidInputError, match="missing"):
        log_potential(cmodel, {"A": 0})
    assert called == []


@pytest.mark.parametrize("decorator", [pyomo_constraint_fn, toulbar2_constraint_fn])
def test_constrained_pyomo_toulbar2_not_supported(decorator):
    @decorator()
    def c(model):
        pass

    cmodel = ConstrainedDiscreteBayesianNetwork(_bn(), c)
    with pytest.raises(NotImplementedError, match="not supported"):
        log_potential(cmodel, {"A": 0, "B": 1})


@require_smoek
def test_constrained_bn_algebraic():
    cmodel = bn_examples.cancer1_BN_constrained_algebraic_conin().pgm
    a = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
    assert log_potential(cmodel, a) == pytest.approx(log_probability(cmodel.model, a))
    assert log_potential(cmodel, {**a, "Xray": 0}) == -math.inf


@require_smoek
def test_constrained_hmm_algebraic():
    cmodel = hmm_examples.create_chmm1_algebraic()
    observed = ["o0"] * 13
    assert log_potential(
        cmodel, ["h0"] * 11 + ["h1"] * 2, observed=observed
    ) == pytest.approx(
        log_probability(cmodel.model, ["h0"] * 11 + ["h1"] * 2, observed=observed)
    )
    assert log_potential(cmodel, ["h0"] * 13, observed=observed) == -math.inf


# ---------------------------------------------------------------------
# pgmpy models
# ---------------------------------------------------------------------


@require_pgmpy
def test_pgmpy_mn():
    pgm = mn_examples.ABC_pgmpy().pgm
    assert log_potential(pgm, {"A": 2, "B": 2, "C": 1}) == pytest.approx(
        2.4849066497880004
    )
    with pytest.raises(InvalidInputError, match="partition function"):
        log_probability(pgm, {"A": 2, "B": 2, "C": 1})


@require_pgmpy
def test_pgmpy_bn():
    pgm = bn_examples.cancer1_BN_pgmpy().pgm
    a = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
    assert log_probability(pgm, a) == pytest.approx(
        math.log(pgm.get_state_probability(a))
    )
    assert log_potential(pgm, a) == pytest.approx(log_probability(pgm, a))


@require_pgmpy
def test_pgmpy_dbn_matches_the_conin_dbn():
    """simple1_DDBN_pgmpy holds the CPD of B for slices 0 and 1; converted, they
    used to become two copies of the template CPD of (B, t), so the unrolled
    network had two CPDs for each (B, t) and log_probability refused it."""
    pgm = dbn_examples.simple1_DDBN_pgmpy().pgm
    conin_pgm = dbn_examples.simple1_DDBN_conin().pgm
    a = {("A", 0): 0, ("A", 1): 0, ("B", 0): 1, ("B", 1): 1}
    assert log_probability(pgm, a) == pytest.approx(log_probability(conin_pgm, a))
