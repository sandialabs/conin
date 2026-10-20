"""Tests of the constrained models' is_feasible methods."""

import pytest

from conin.constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.feasibility import mvr_accepts
from conin.constraints import (
    FactorOracleConstraint,
    OracleConstraint,
    mvr_constraint_fn,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.constraints.mvr import (
    HomMVR,
    InhomMVR,
    mvr_forbid_state,
    mvr_timerange,
    mvr_visit_state,
)
from conin.exceptions import InvalidInputError
from conin.util import try_import

import conin.examples.bayesian_network as bn_examples
import conin.examples.dynamic_bayesian_network as dbn_examples
import conin.examples.hidden_markov_model as hmm_examples
import conin.examples.markov_network as mn_examples

with try_import() as smoek_available:
    import smoek  # noqa: F401

require_smoek = pytest.mark.skipif(not smoek_available, reason="smoek not installed")


# ---------------------------------------------------------------------
# MN / BN / DBN
# ---------------------------------------------------------------------


def test_mn_oracle_and_factor_oracle():
    mn = mn_examples.ABC_conin().pgm
    for c in [
        OracleConstraint(func=lambda s: s["A"] != s["B"]),
        FactorOracleConstraint(func=lambda s: s["A"] != s["B"], nodes=["A", "B"]),
    ]:
        cmodel = ConstrainedDiscreteMarkovNetwork(mn, c)
        assert cmodel.is_feasible({"A": 0, "B": 1, "C": 0})
        assert not cmodel.is_feasible({"A": 1, "B": 1, "C": 0})


def test_factor_oracle_sees_only_its_scope():
    seen = []

    def func(states):
        seen.append(dict(states))
        return True

    mn = mn_examples.ABC_conin().pgm
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mn, FactorOracleConstraint(func=func, nodes=["C", "A"])
    )
    assert cmodel.is_feasible({"A": 0, "B": 1, "C": 2})
    assert seen == [{"C": 2, "A": 0}]


def test_all_constraints_must_hold():
    mn = mn_examples.ABC_conin().pgm
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mn,
        [
            OracleConstraint(func=lambda s: s["A"] == 0),
            OracleConstraint(func=lambda s: s["B"] == 0),
        ],
    )
    assert cmodel.is_feasible({"A": 0, "B": 0, "C": 0})
    assert not cmodel.is_feasible({"A": 0, "B": 1, "C": 0})
    assert not cmodel.is_feasible({"A": 1, "B": 0, "C": 0})


def test_bn_example_solution_is_feasible():
    example = bn_examples.cancer1_BN_constrained_oracle_conin()
    assert example.pgm.is_feasible(example.solutions[0].states)


def test_dbn_example_solution_is_feasible():
    example = dbn_examples.simple1_DDBN_constrained_oracle_conin()
    assert example.pgm.is_feasible(example.solutions[0].states)
    states = dict(example.solutions[0].states)
    states["B", 1] = 1 - states["B", 1]
    assert not example.pgm.is_feasible(states)


def test_dbn_callable_scope_sees_data_T():
    """Callable node scopes see data = Munch(T), as for variable elimination."""
    cmodel = dbn_examples.simple1_DDBN_constrained_oracle_conin().pgm
    a = {(n, t): 0 for n in "AB" for t in range(3)}
    assert cmodel.is_feasible(a)
    a["A", 2] = 1
    # The predicate only compares t=0 and t=1.
    assert cmodel.is_feasible(a)
    a["A", 1] = 1
    assert not cmodel.is_feasible(a)


def test_factor_oracle_scope_outside_assignment():
    mn = mn_examples.ABC_conin().pgm
    cmodel = ConstrainedDiscreteMarkovNetwork(
        mn, FactorOracleConstraint(func=lambda s: True, nodes=["A", "D"])
    )
    with pytest.raises(InvalidInputError, match="not in the assignment"):
        cmodel.is_feasible({"A": 0, "B": 0, "C": 0})


@pytest.mark.parametrize("decorator", [pyomo_constraint_fn, toulbar2_constraint_fn])
def test_pyomo_toulbar2_not_supported(decorator):
    @decorator()
    def c(model):
        pass

    pgm = bn_examples.cancer1_BN_conin().pgm
    cmodel = ConstrainedDiscreteBayesianNetwork(pgm, c)
    a = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
    with pytest.raises(NotImplementedError, match="optimization feasibility"):
        cmodel.is_feasible(a)


def test_bad_assignment():
    cmodel = ConstrainedDiscreteBayesianNetwork(bn_examples.cancer1_BN_conin().pgm)
    with pytest.raises(InvalidInputError, match="missing"):
        cmodel.is_feasible({"Cancer": 1})


@require_smoek
def test_bn_algebraic():
    cmodel = bn_examples.cancer1_BN_constrained_algebraic_conin().pgm
    a = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
    assert cmodel.is_feasible(a)
    assert not cmodel.is_feasible({**a, "Xray": 0})


@require_smoek
def test_dbn_algebraic():
    cmodel = dbn_examples.simple1_DDBN_constrained_algebraic_conin().pgm
    a = {("A", 0): 0, ("A", 1): 0, ("B", 0): 1, ("B", 1): 1}
    assert cmodel.is_feasible(a)
    assert not cmodel.is_feasible({**a, ("A", 1): 1})


# ---------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [hmm_examples.create_chmm1_oracle, hmm_examples.create_chmm1_oracle_ve],
    ids=["oracle", "factor_oracle"],
)
def test_hmm_oracle_and_factor_oracle(factory):
    """ConstrainedHiddenMarkovModel.is_feasible used to call every constraint
    with {t: h}, which is wrong for factor oracles (their nodes are ("H", t))."""
    cmodel = factory()
    assert not cmodel.is_feasible(["h0"] * 9)
    assert cmodel.is_feasible(["h0"] * 10)
    assert cmodel.is_feasible(["h0"] * 12 + ["h1"])
    assert not cmodel.is_feasible(["h0"] * 13)


def test_hmm_dict_sequence():
    cmodel = hmm_examples.create_chmm1_oracle()
    assert cmodel.is_feasible({t: "h0" for t in range(11)})


def test_hmm_factor_oracle_on_observed_nodes():
    def nodes(data):
        for t in data.hmm.T:
            yield ("H", t)
            yield ("E", t)

    c = FactorOracleConstraint(
        func=lambda s: all(
            (s["H", t] == "h0") == (s["E", t] == "o0") for t in range(len(s) // 2)
        ),
        nodes=nodes,
    )
    cmodel = ConstrainedHiddenMarkovModel(hmm_examples.create_hmm1(), c)
    assert cmodel.is_feasible(["h0", "h1"], ["o0", "o1"])
    assert not cmodel.is_feasible(["h0", "h1"], ["o0", "o0"])
    with pytest.raises(InvalidInputError, match="not in the assignment"):
        cmodel.is_feasible(["h0", "h1"])


def test_hmm_mvr():
    """ConstrainedHiddenMarkovModel.is_feasible used to call MVRs with {t: h}."""
    hmm = hmm_examples.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm, [mvr_forbid_state(hmm, ["h1"])])
    assert cmodel.is_feasible(["h0"] * 5)
    assert not cmodel.is_feasible(["h0", "h1", "h0"])


def test_hmm_mvr_constraint_factory_sees_data():
    seen = []

    @mvr_constraint_fn()
    def visit(hmm, data):
        seen.append(data)
        return mvr_visit_state(hmm, ["h1"])

    cmodel = ConstrainedHiddenMarkovModel(
        hmm_examples.create_hmm1(), visit, data={"x": 1}
    )
    assert cmodel.is_feasible(["h0", "h1"])
    assert not cmodel.is_feasible(["h0", "h0"])
    assert seen == [{"x": 1}, {"x": 1}]


def test_hmm_mvr_time_range():
    hmm = hmm_examples.create_hmm1()
    mvr = mvr_timerange(mvr_forbid_state(hmm, ["h1"]), [1, 2], inplace=False)
    cmodel = ConstrainedHiddenMarkovModel(hmm, [mvr])
    assert cmodel.is_feasible(["h1", "h0", "h0", "h1"])
    assert not cmodel.is_feasible(["h0", "h0", "h1", "h0"])
    with pytest.raises(InvalidInputError, match="exceeds"):
        cmodel.is_feasible(["h0", "h0"])


def test_hmm_pyomo_not_supported():
    with pytest.raises(NotImplementedError):
        hmm_examples.create_chmm1_pyomo().is_feasible(["h0"] * 3)


def test_hmm_bad_sequence():
    cmodel = hmm_examples.create_chmm1_oracle()
    with pytest.raises(InvalidInputError, match="not a hidden state"):
        cmodel.is_feasible(["hx"])


@require_smoek
def test_hmm_algebraic():
    cmodel = hmm_examples.create_chmm1_algebraic()
    assert not cmodel.is_feasible(["h0"] * 9)
    assert cmodel.is_feasible(["h0"] * 10)
    assert not cmodel.is_feasible(["h0"] * 13)


# ---------------------------------------------------------------------
# mvr_accepts
# ---------------------------------------------------------------------


def _parity_mvr(time_range=None):
    """Accepts iff the window has an even number of "b"s."""
    return HomMVR(
        hidden_states=["a", "b"],
        mediation_states=["even", "odd"],
        ini={"a": "even", "b": "odd"},
        upd={
            ("even", "a"): "even",
            ("even", "b"): "odd",
            ("odd", "a"): "odd",
            ("odd", "b"): "even",
        },
        evl={"even": True, "odd": False},
        time_range=time_range,
    )


def test_mvr_accepts_hom():
    mvr = _parity_mvr()
    assert mvr_accepts(mvr, ["a", "a"])
    assert mvr_accepts(mvr, ["b", "a", "b"])
    assert not mvr_accepts(mvr, ["b", "a", "a"])


def test_mvr_accepts_window():
    mvr = _parity_mvr([1, 2])
    assert mvr_accepts(mvr, ["b", "b", "b", "a"])
    assert not mvr_accepts(mvr, ["a", "b", "a", "a"])


def test_mvr_accepts_inhom():
    """Accepts iff the last state of the window is "b"."""
    mvr = InhomMVR(
        hidden_states=["a", "b"],
        mediation_states=[[f"t{t}a", f"t{t}b"] for t in range(3)],
        ini={"a": "t0a", "b": "t0b"},
        upd=[
            {(m, h): f"t{t + 1}{h}" for m in [f"t{t}a", f"t{t}b"] for h in "ab"}
            for t in range(2)
        ],
        evl=[{f"t{t}a": False, f"t{t}b": True} for t in range(3)],
    )
    assert mvr_accepts(mvr, ["a", "a", "b"])
    assert not mvr_accepts(mvr, ["b", "b", "a"])
    with pytest.raises(InvalidInputError, match="too short"):
        mvr_accepts(mvr, ["a"] * 4)


def test_mvr_accepts_unknown_state():
    with pytest.raises(InvalidInputError, match="not one of the MVR"):
        mvr_accepts(_parity_mvr(), ["a", "c"])
