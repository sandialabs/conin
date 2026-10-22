"""Tests of conin.scoring.log_probability_query.

The values are checked against brute-force enumeration: the sum of
``exp(log_potential)`` over the total assignments that agree with the
assignment and the evidence (the feasible ones, for constrained models).
"""

import importlib
import itertools
import math

import munch
import pytest

import conin.examples.bayesian_network as bex
import conin.examples.dynamic_bayesian_network as dex
import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import FactorOracleConstraint, algebraic_constraint_fn
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    HiddenMarkovModel,
)
from conin.representations.pgmpy import (
    PgmpyBNRepresentation,
    PgmpyDBNRepresentation,
    PgmpyHMMRepresentation,
    PgmpyMNRepresentation,
)
from conin.scoring import log_potential, log_probability, log_probability_query
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as smoek_available:
    import smoek  # noqa: F401

require_pgmpy = pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
require_smoek = pytest.mark.skipif(not smoek_available, reason="smoek not installed")

lpq = importlib.import_module("conin.scoring.log_probability_query")

TOL = 1e-9


# ---------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------


def _total_weights(model, stop=None):
    """Yield ``(total assignment, exp(log_potential))`` over every total assignment.

    DBN assignments are over ``0..stop``; HMM assignments are keyed by the
    unrolled nodes ``("H", t)`` and ``("E", t)`` for ``t`` in ``0..stop``.
    """
    base = getattr(model, "model", model)
    if isinstance(base, HiddenMarkovModel):
        T = stop + 1
        for hidden in itertools.product(base.hidden_states, repeat=T):
            for observed in itertools.product(base.observed_states, repeat=T):
                lp = log_potential(model, list(hidden), observed=list(observed))
                a = {("H", t): hidden[t] for t in range(T)}
                a.update({("E", t): observed[t] for t in range(T)})
                yield a, (math.exp(lp) if lp > -math.inf else 0.0)
        return
    if isinstance(base, DynamicDiscreteBayesianNetwork):
        states = {
            (name, t): s
            for name, s in base.dynamic_states.items()
            for t in range(stop + 1)
        }
    else:
        states = base.states
    nodes = list(states)
    for values in itertools.product(*(states[n] for n in nodes)):
        a = dict(zip(nodes, values))
        lp = log_potential(model, a)
        yield a, (math.exp(lp) if lp > -math.inf else 0.0)


def _brute_force(model, assignment, evidence=None, stop=None):
    evidence = evidence or {}
    num = den = 0.0
    for a, w in _total_weights(model, stop):
        if all(a[k] == v for k, v in evidence.items()):
            den += w
            if all(a[k] == v for k, v in assignment.items()):
                num += w
    return math.log(num) - math.log(den) if num > 0 else -math.inf


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def _mn():
    return mex.ABC_conin().pgm


def _mn_sparse():
    """A Markov network whose factors are not uniform, with a default value."""
    return DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1, 2], "C": [0, 1]},
        factors=[
            DiscreteFactor(nodes=["A", "B"], values=[1, 2, 3, 4, 5, 6]),
            DiscreteFactor(nodes=["B", "C"], values=[2, 1, 1, 3, 0.5, 2]),
            DiscreteFactor(nodes=["C"], values=[1, 4]),
        ],
    )


def _bn():
    return bex.cancer1_BN_conin().pgm


def _dbn():
    return dex.simple1_DDBN_conin().pgm


def _hmm():
    return hex_.create_hmm1()


def _hmm_constraint():
    """At most one ``h1`` over the HMM's time steps."""

    def nodes(data):
        return [("H", t) for t in data.hmm.T]

    return FactorOracleConstraint(
        func=lambda states: sum(v == "h1" for v in states.values()) <= 1,
        nodes=nodes,
        name="at_most_one_h1",
    )


def _cmn():
    return ConstrainedDiscreteMarkovNetwork(
        _mn_sparse(),
        FactorOracleConstraint(
            func=lambda s: s["A"] != s["C"], nodes=["A", "C"], name="a_ne_c"
        ),
    )


def _cbn():
    return bex.cancer1_BN_constrained_oracle_conin().pgm


def _cdbn():
    return dex.simple1_DDBN_constrained_oracle_conin().pgm


def _chmm():
    return ConstrainedHiddenMarkovModel(_hmm(), _hmm_constraint())


MN_CASES = [
    ({"A": 2}, None),
    ({"A": 0, "B": 1}, None),
    ({"A": 1}, {"C": 0}),
    ({"A": 0, "B": 2, "C": 1}, None),
    ({"B": 1}, {"A": 0, "C": 1}),
]
BN_CASES = [
    ({"Xray": 1}, None),
    ({"Cancer": 1}, {"Xray": 1, "Dyspnoea": 1}),
    ({"Smoker": 0, "Pollution": 1}, {"Cancer": 0}),
    ({"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}, None),
]
DBN_CASES = [
    ({("A", 1): 1}, None),
    ({("B", 2): 0}, {("A", 0): 1}),
    ({("A", 0): 0, ("A", 2): 1}, {("B", 1): 1}),
]
HMM_CASES = [
    ({("H", 2): "h1"}, None),
    ({("H", 1): "h0"}, {("E", 0): "o1", ("E", 1): "o0", ("E", 2): "o0"}),
    ({("E", 0): "o0", ("E", 1): "o1", ("E", 2): "o0"}, None),
    ({("H", 0): "h1", ("H", 2): "h0"}, {("E", 1): "o1"}),
]


def _params(model, cases, stop=None):
    return [
        pytest.param(model, a, e, stop, id=f"{model.__name__}-{i}")
        for i, (a, e) in enumerate(cases)
    ]


BRUTE_FORCE = (
    _params(_mn, MN_CASES[:3])
    + _params(_mn_sparse, MN_CASES[1:])
    + _params(_bn, BN_CASES)
    + _params(_dbn, DBN_CASES, stop=2)
    + _params(_hmm, HMM_CASES, stop=2)
)
CONSTRAINED = (
    _params(_cmn, MN_CASES[1:4] + [({"B": 1}, {"A": 0})])
    + _params(
        _cbn,
        [
            ({"Xray": 1}, None),
            ({"Cancer": 1}, {"Xray": 1, "Smoker": 1}),
            ({"Smoker": 0, "Pollution": 1}, {"Dyspnoea": 0}),
            (
                {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1},
                None,
            ),
        ],
    )
    + _params(_cdbn, [({("A", 1): 1}, None), ({("B", 0): 1}, {("A", 0): 0})], stop=1)
    + _params(_chmm, HMM_CASES, stop=2)
)


# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------


@require_pgmpy
@pytest.mark.parametrize("model,assignment,evidence,stop", BRUTE_FORCE)
def test_matches_brute_force(model, assignment, evidence, stop):
    pgm = model()
    value = log_probability_query(pgm, assignment, evidence=evidence, stop=stop)
    assert value == pytest.approx(
        _brute_force(pgm, assignment, evidence, stop), abs=TOL
    )


@require_pgmpy
@pytest.mark.parametrize("model,assignment,evidence,stop", CONSTRAINED)
def test_constrained_matches_brute_force(model, assignment, evidence, stop):
    """Under constraints: the brute force sums over the feasible assignments only."""
    cmodel = model()
    value = log_probability_query(cmodel, assignment, evidence=evidence, stop=stop)
    expected = _brute_force(cmodel, assignment, evidence, stop)
    assert value == pytest.approx(expected, abs=TOL)
    # The constraints change the value (the cases are chosen so).
    unconstrained = _brute_force(cmodel.model, assignment, evidence, stop)
    assert abs(expected - unconstrained) > 1e-6


@require_pgmpy
@require_smoek
@pytest.mark.parametrize(
    "example,assignment,evidence,stop",
    [
        (bex.cancer1_BN_constrained_algebraic_conin, {"Xray": 1}, None, None),
        (bex.cancer1_BN_constrained_algebraic_conin, {"Cancer": 1}, {"Xray": 0}, None),
        (dex.simple1_DDBN_constrained_algebraic_conin, {("A", 1): 1}, None, 1),
    ],
    ids=["bn", "bn-evidence", "dbn"],
)
def test_algebraic_constraints_match_brute_force(example, assignment, evidence, stop):
    cmodel = example().pgm
    value = log_probability_query(cmodel, assignment, evidence=evidence, stop=stop)
    assert value == pytest.approx(
        _brute_force(cmodel, assignment, evidence, stop), abs=TOL
    )


@require_pgmpy
@require_smoek
def test_algebraic_hmm_constraints_match_brute_force():
    @algebraic_constraint_fn()
    def at_least_two_h0(M, D):
        return sum(M.V("H", t, "h0") for t in D.hmm.T) >= 2

    cmodel = ConstrainedHiddenMarkovModel(_hmm(), at_least_two_h0)
    assignment, evidence = {("H", 0): "h1"}, {("E", 2): "o1"}
    value = log_probability_query(cmodel, assignment, evidence=evidence, stop=2)
    assert value == pytest.approx(
        _brute_force(cmodel, assignment, evidence, stop=2), abs=TOL
    )


@require_pgmpy
@pytest.mark.parametrize(
    "model,assignment,observed",
    [
        (_bn, {"Cancer": 0, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 0}, None),
        (_dbn, {("A", 0): 0, ("A", 1): 1, ("A", 2): 1, ("B", 0): 1, ("B", 1): 0, ("B", 2): 1}, None),
        (_hmm, ["h0", "h1", "h1"], ["o0", "o1", "o0"]),
    ],
    ids=["bn", "dbn", "hmm"],
)  # fmt: skip
def test_total_assignment_equals_log_probability(model, assignment, observed):
    pgm = model()
    expected = log_probability(pgm, assignment, observed=observed)
    if observed is not None:
        T = len(assignment)
        assignment = {("H", t): assignment[t] for t in range(T)}
        assignment.update({("E", t): observed[t] for t in range(T)})
    assert log_probability_query(pgm, assignment) == pytest.approx(expected, abs=TOL)


@require_pgmpy
def test_mn_total_assignment_is_normalised():
    """For a Markov network the partition function cancels: log_potential - log Z."""
    mn = _mn_sparse()
    log_Z = math.log(sum(w for _, w in _total_weights(mn)))
    x = {"A": 1, "B": 2, "C": 0}
    assert log_probability_query(mn, x) == pytest.approx(
        log_potential(mn, x) - log_Z, abs=TOL
    )


@require_pgmpy
@pytest.mark.parametrize(
    "model,assignment,evidence",
    [
        (_mn, {"A": 1}, {"B": 2, "C": 0}),
        (_bn, {"Cancer": 1}, {"Xray": 1, "Smoker": 0}),
        (_cbn, {"Cancer": 1}, {"Xray": 1}),
        (_chmm, {("H", 1): "h1"}, {("E", 0): "o1", ("E", 1): "o1"}),
    ],
    ids=["mn", "bn", "constrained-bn", "constrained-hmm"],
)
def test_conditional_is_joint_minus_marginal(model, assignment, evidence):
    pgm = model()
    conditional = log_probability_query(pgm, assignment, evidence=evidence, stop=2)
    joint = log_probability_query(pgm, {**assignment, **evidence}, stop=2)
    marginal = log_probability_query(pgm, evidence, stop=2)
    assert conditional == pytest.approx(joint - marginal, abs=TOL)


@require_pgmpy
def test_under_constraints_is_joint_minus_constraint_probability():
    """``log P(x | C) = log P(x, C) - log P(C)``, from queries on the unconstrained model.

    The constraint of ``cancer1_BN_constrained_oracle_conin`` is ``Dyspnoea != Xray``.
    """
    cmodel = _cbn()
    bn = cmodel.model

    def p(assignment):
        return math.exp(log_probability_query(bn, assignment))

    p_C = p({"Dyspnoea": 0, "Xray": 1}) + p({"Dyspnoea": 1, "Xray": 0})
    p_x_C = p({"Cancer": 1, "Dyspnoea": 0, "Xray": 1}) + p(
        {"Cancer": 1, "Dyspnoea": 1, "Xray": 0}
    )
    assert log_probability_query(cmodel, {"Cancer": 1}) == pytest.approx(
        math.log(p_x_C) - math.log(p_C), abs=TOL
    )


@require_pgmpy
def test_hmm_likelihood_of_the_observations():
    """``log P(observations)``, the forward-algorithm likelihood."""
    hmm = _hmm()
    observed = ["o0", "o1", "o1", "o0"]
    expected = math.log(
        sum(
            math.exp(log_probability(hmm, list(h), observed=observed))
            for h in itertools.product(hmm.hidden_states, repeat=len(observed))
        )
    )
    assignment = {("E", t): o for t, o in enumerate(observed)}
    assert log_probability_query(hmm, assignment) == pytest.approx(expected, abs=TOL)


@require_pgmpy
def test_bare_and_unconstrained_models_agree():
    a, e = {"Cancer": 1}, {"Xray": 1}
    assert log_probability_query(_bn(), a, evidence=e) == pytest.approx(
        log_probability_query(ConstrainedDiscreteBayesianNetwork(_bn()), a, evidence=e),
        abs=0,
    )


@require_pgmpy
def test_pgmpy_models_are_converted():
    a, e = {"Cancer": 1}, {"Xray": 1}
    pgmpy_model = bex.cancer1_BN_pgmpy().pgm
    assert log_probability_query(pgmpy_model, a, evidence=e) == pytest.approx(
        log_probability_query(_bn(), a, evidence=e), abs=TOL
    )


@require_pgmpy
def test_zero_probability_assignment_is_minus_inf():
    hmm0 = hex_.create_hmm0()  # h0 then h1 forever; h0 emits o0, h1 emits o1
    assert log_probability_query(hmm0, {("H", 1): "h0"}) == -math.inf
    assert (
        log_probability_query(hmm0, {("H", 0): "h1"}, evidence={("E", 1): "o1"})
        == -math.inf
    )


@require_pgmpy
def test_stop_default_and_a_later_stop():
    """``stop`` defaults to the largest time; a later one gives the same unconstrained value."""
    hmm = _hmm()
    a = {("H", 1): "h1"}
    assert log_probability_query(hmm, a) == pytest.approx(
        log_probability_query(hmm, a, stop=3), abs=TOL
    )
    dbn = _dbn()
    a = {("A", 0): 1}  # stop defaults to start + 1
    assert log_probability_query(dbn, a) == pytest.approx(math.log(0.1), abs=TOL)
    assert log_probability_query(dbn, a, stop=3) == pytest.approx(
        math.log(0.1), abs=TOL
    )


@require_pgmpy
def test_the_horizon_matters_under_constraints():
    """The HMM constraint (at most one h1) covers the times start..stop."""
    cmodel = _chmm()
    a = {("H", 0): "h1"}
    short = log_probability_query(cmodel, a, stop=1)
    long = log_probability_query(cmodel, a, stop=2)
    assert short == pytest.approx(_brute_force(cmodel, a, stop=1), abs=TOL)
    assert long == pytest.approx(_brute_force(cmodel, a, stop=2), abs=TOL)
    assert abs(short - long) > 1e-6


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


@require_pgmpy
def test_zero_probability_evidence_raises():
    hmm0 = hex_.create_hmm0()
    with pytest.raises(InvalidInputError, match="the evidence has probability zero"):
        log_probability_query(hmm0, {("E", 0): "o0"}, evidence={("H", 0): "h1"})


@require_pgmpy
def test_zero_probability_constraints_raise():
    """P(C) = 0: the example's constraints need ten or more h0 over the time steps."""
    cmodel = hex_.create_chmm1_oracle_ve()
    with pytest.raises(
        InvalidInputError, match="the constraints have probability zero"
    ):
        log_probability_query(cmodel, {("H", 0): "h0"}, stop=2)
    with pytest.raises(
        InvalidInputError,
        match="the evidence and the constraints have probability zero",
    ):
        log_probability_query(
            cmodel, {("H", 0): "h0"}, evidence={("E", 0): "o0"}, stop=2
        )


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unsupported log-probability method: viterbi"):
        log_probability_query(_bn(), {"Xray": 1}, method="viterbi")


@pytest.mark.parametrize("model", ["not_a_model", 3, object()])
def test_unsupported_model_type_raises_type_error(model):
    with pytest.raises(TypeError, match="Unsupported model type"):
        log_probability_query(model, {"A": 0})


@pytest.mark.parametrize(
    "example",
    [
        lambda: bex.cancer1_BN_constrained_pyomo_conin().pgm,
        lambda: bex.cancer1_BN_constrained_toulbar2_conin().pgm,
        lambda: dex.simple1_DDBN_constrained_pyomo_conin().pgm,
        hex_.create_chmm1_oracle,
        hex_.create_chmm1_pyomo,
        lambda: hex_.create_chmm1_recursive(),
    ],
    ids=[
        "bn-pyomo",
        "bn-toulbar2",
        "dbn-pyomo",
        "hmm-oracle",
        "hmm-pyomo",
        "hmm-recursive",
    ],
)
def test_unsupported_families_raise_the_registry_error(example):
    cmodel = example()
    var = ("H", 0) if isinstance(cmodel, ConstrainedHiddenMarkovModel) else "Xray"
    if isinstance(cmodel, ConstrainedDynamicDiscreteBayesianNetwork):
        var = ("A", 0)
    with pytest.raises(
        InvalidInputError, match="No .pgmpy. representation is available"
    ):
        log_probability_query(cmodel, {var: 0})


@pytest.mark.parametrize(
    "assignment,evidence,match",
    [
        ({}, None, "at least one variable"),
        ([("H", 0), "h0"], None, "assignment of log_probability_query must be a dict"),
        ({("H", 0): "h0"}, ["o0", "o1"], "evidence of log_probability_query must be a dict"),
        ({("H", 0): "h0"}, {("H", 0): "h0"}, "both the assignment and the evidence"),
        ({("X", 0): "h0"}, None, r"Unknown variable \('X', 0\) in the assignment"),
        ({("H", 0): "h9"}, None, r"Unknown state 'h9' of \('H', 0\) in the assignment"),
        ({("H", 0): "h0"}, {("E", 0): "o9"}, r"Unknown state 'o9' of \('E', 0\) in the evidence"),
        ({("H", 3): "h0"}, None, None),
    ],
    ids=["empty", "list", "list-evidence", "overlap", "variable", "state", "evidence-state", "ok"],
)  # fmt: skip
def test_bad_input_raises(assignment, evidence, match):
    if match is None:
        pytest.importorskip("pgmpy")
        assert log_probability_query(_hmm(), assignment, evidence=evidence) < 0
        return
    with pytest.raises(InvalidInputError, match=match):
        log_probability_query(_hmm(), assignment, evidence=evidence)


def test_a_variable_past_stop_is_unknown():
    with pytest.raises(InvalidInputError, match=r"Unknown variable \('H', 3\)"):
        log_probability_query(_hmm(), {("H", 3): "h0"}, stop=2)


@pytest.mark.parametrize("start,stop", [(0, 0), (2, 1)])
def test_stop_must_be_after_start(start, stop):
    with pytest.raises(InvalidInputError, match="start < stop"):
        log_probability_query(_dbn(), {("A", 0): 0}, start=start, stop=stop)


def test_unknown_options_raise_type_error():
    with pytest.raises(TypeError, match="Unexpected options .*: solver"):
        log_probability_query(_bn(), {"Xray": 1}, solver="glpk")


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------


def test_methods():
    assert lpq.METHODS == ("variable_elimination",)
    assert [lpq._representation_kind(m) for m in lpq.METHODS] == ["pgmpy"]


@pytest.mark.parametrize(
    "model,variables,start,stop,expected",
    [
        (_mn, ["A"], 0, None, {}),
        (_bn, ["Xray"], 0, 5, {}),
        (_dbn, [("A", 0)], 0, None, dict(start=0, stop=1)),
        (_dbn, [("A", 0), ("B", 4)], 0, None, dict(start=0, stop=4)),
        (_dbn, [("A", 0)], 0, 3, dict(start=0, stop=3)),
        (_hmm, [("H", 2), ("E", 5)], 0, None, dict(start=0, stop=5)),
        (_hmm, [("H", 0)], 1, None, dict(start=1, stop=2)),
    ],
)
def test_structure_params(model, variables, start, stop, expected):
    cmodel = lpq.as_constrained_model(model())
    params = lpq._structure_params(cmodel, variables=variables, start=start, stop=stop)
    assert params == expected


@pytest.mark.parametrize(
    "cls",
    [
        PgmpyMNRepresentation,
        PgmpyBNRepresentation,
        PgmpyDBNRepresentation,
        PgmpyHMMRepresentation,
    ],
)
def test_one_overload_per_representation_class(cls):
    resolved = lpq._solve_log_probability.resolve(cls, munch.Munch)
    assert resolved.__annotations__["repn"] is cls


@pytest.mark.parametrize(
    "model,var",
    [(_mn, "A"), (_bn, "Xray"), (_dbn, ("A", 0)), (_hmm, ("H", 0))],
    ids=["mn", "bn", "dbn", "hmm"],
)
def test_each_model_type_builds_its_factor_representation(model, var, monkeypatch):
    seen = []
    monkeypatch.setattr(
        lpq,
        "_log_probability_variable_elimination",
        lambda repn, **kwds: seen.append(type(repn)) or 0.0,
    )
    pgm = model()
    state = {"A": 0, "Xray": 0, ("A", 0): 0, ("H", 0): "h0"}[var]
    assert log_probability_query(pgm, {var: state}) == 0.0
    expected = {
        "A": PgmpyMNRepresentation,
        "Xray": PgmpyBNRepresentation,
        ("A", 0): PgmpyDBNRepresentation,
        ("H", 0): PgmpyHMMRepresentation,
    }[var]
    assert seen == [expected]
