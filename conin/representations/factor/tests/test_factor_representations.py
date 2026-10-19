"""Tests for the "factor" representations and variable elimination (PR 4e)."""

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
from conin.constraints import (
    FactorOracleConstraint,
    oracle_constraint_fn,
    pyomo_constraint_fn,
)
from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    HiddenMarkovModel,
    create_bn_from_dbn,
    create_dbn_from_hmm,
)
from conin.inference import map_query
from conin.representations import supported_representations
from conin.representations.factor import (
    FactorBNRepresentation,
    FactorDBNRepresentation,
    FactorHMMRepresentation,
    FactorMNRepresentation,
)
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.models

skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)


def _abc_all_different():
    @oracle_constraint_fn(nodes=["A", "B", "C"])
    def all_different(states):
        return len(set(states.values())) == 3

    return all_different


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type",
    [
        DiscreteMarkovNetwork,
        DiscreteBayesianNetwork,
        DynamicDiscreteBayesianNetwork,
        HiddenMarkovModel,
    ],
)
def test_factor_registered_families(model_type):
    pairs = supported_representations(model_type)
    for family in (None, "factor_oracle", "algebraic"):
        assert ("factor", family) in pairs
    for family in ("pyomo", "toulbar2", "oracle", "mvr"):
        assert ("factor", family) not in pairs


def test_factor_rejects_pyomo_constraints():
    @pyomo_constraint_fn()
    def noop(M, D):
        pass

    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm, noop)
    with pytest.raises(InvalidInputError, match="No 'factor' representation"):
        cmodel.representation("factor")


def test_variable_elimination_rejects_pyomo_constraints():
    """Was a TypeError from the old VE constraint helper."""
    example = bex.cancer1_BN_constrained_pyomo_conin()
    with pytest.raises(InvalidInputError, match="No 'factor' representation"):
        map_query(example.pgm, method="variable_elimination")


@pytest.mark.parametrize(
    "cmodel,params,cls",
    [
        (
            ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm),
            {},
            FactorMNRepresentation,
        ),
        (
            ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm),
            {},
            FactorBNRepresentation,
        ),
        (
            ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm),
            dict(stop=4),
            FactorDBNRepresentation,
        ),
        (
            ConstrainedHiddenMarkovModel(hex_.create_hmm1()),
            dict(stop=4),
            FactorHMMRepresentation,
        ),
    ],
)
def test_factor_representation_classes(cmodel, params, cls):
    repn = cmodel.representation("factor", **params)
    assert type(repn) is cls
    assert repn.kind == "factor"
    # Unconstrained: the VE model is the (unrolled) model itself.
    assert repn.ve_model is repn.pgm
    assert repn.constraint_evidence == {}


# ---------------------------------------------------------------------
# Structure params
# ---------------------------------------------------------------------


def test_dbn_structure_params():
    dbn = dex.weather_conin().pgm
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
    default = cmodel.representation("factor")
    assert (default.start, default.stop) == (0, 1)
    # (start > 0 fails in create_bn_from_dbn for every backend; §9.)
    repn = cmodel.representation("factor", start=0, stop=3)
    assert (repn.start, repn.stop) == (0, 3)
    assert repn.constraint_data() == munch.Munch(T=[0, 1, 2, 3])
    assert set(repn.nodes) == set(create_bn_from_dbn(dbn=dbn, start=0, stop=3).nodes)


def test_hmm_structure_params():
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm)
    with pytest.raises(AssertionError, match="Expecting evidence"):
        cmodel.representation("factor")
    repn = cmodel.representation("factor", stop=3)
    assert (repn.start, repn.stop) == (0, 3)
    assert repn.constraint_data() == munch.Munch(hmm=munch.Munch(T=[0, 1, 2, 3]))
    unrolled = create_bn_from_dbn(dbn=create_dbn_from_hmm(hmm), start=0, stop=3)
    assert set(repn.nodes) == set(unrolled.nodes)


# ---------------------------------------------------------------------
# The constraint data each model passes (unchanged from before PR 4e)
# ---------------------------------------------------------------------


def _recording_constraint(seen, nodes):
    def scope(data):
        seen.append(data)
        return nodes

    return FactorOracleConstraint(func=lambda s: True, nodes=scope, name="rec")


def test_constraint_data_mn_bn_is_none():
    seen = []
    ConstrainedDiscreteMarkovNetwork(
        mex.ABC_conin().pgm, _recording_constraint(seen, ["A"])
    ).representation("factor")
    ConstrainedDiscreteBayesianNetwork(
        bex.cancer1_BN_conin().pgm, _recording_constraint(seen, ["Smoker"])
    ).representation("factor")
    assert seen == [None, None]


def test_constraint_data_dbn_and_hmm():
    seen = []
    ConstrainedDynamicDiscreteBayesianNetwork(
        dex.simple1_DDBN_conin().pgm, _recording_constraint(seen, [("A", 0)])
    ).representation("factor", stop=2)
    ConstrainedHiddenMarkovModel(
        hex_.create_hmm1(), _recording_constraint(seen, [("H", 0)])
    ).representation("factor", stop=2)
    assert seen == [munch.Munch(T=[0, 1, 2]), munch.Munch(hmm=munch.Munch(T=[0, 1, 2]))]


def test_constraints_are_materialised_at_build_time():
    seen = []
    cmodel = ConstrainedDiscreteBayesianNetwork(
        bex.cancer1_BN_conin().pgm, _recording_constraint(seen, ["Smoker"])
    )
    repn = cmodel.representation("factor")
    assert len(seen) == 1
    assert repn.constraint_evidence == {("rec", -1): 1}
    assert ("rec", -1) in repn.ve_model.nodes
    assert ("rec", -1) not in repn.nodes
    repn.query_evidence({"Smoker": 0})
    assert len(seen) == 1  # queries don't re-materialise


# ---------------------------------------------------------------------
# Evidence and queries don't mutate the representation
# ---------------------------------------------------------------------


def test_query_evidence_does_not_mutate_the_repn_or_the_input():
    cmodel = bex.cancer1_BN_constrained_oracle_conin().pgm
    repn = cmodel.representation("factor")
    before = dict(repn.constraint_evidence)
    user = {"Smoker": 0}
    q1 = repn.query_evidence(user)
    assert q1 == {"Smoker": 0, **before}
    q1["Cancer"] = 1
    assert user == {"Smoker": 0}
    assert repn.constraint_evidence == before
    assert repn.query_evidence() == before
    assert repn.query_evidence() is not repn.constraint_evidence


@skipif_pgmpy_not_available
def test_two_create_model_calls_give_independent_models():
    repn = bex.cancer1_BN_constrained_oracle_conin().pgm.representation("factor")
    m1 = repn.create_model()
    m2 = repn.create_model()
    assert m1 is not m2
    assert isinstance(m1, pgmpy.models.DiscreteBayesianNetwork)
    m1.remove_node("Xray")
    assert "Xray" in m2.nodes()
    assert "Xray" in repn.create_model().nodes()


@skipif_pgmpy_not_available
def test_create_model_mn_is_a_pgmpy_markov_network():
    repn = mex.ABC_constrained_oracle_conin().pgm.representation("factor")
    assert isinstance(repn.create_model(), pgmpy.models.DiscreteMarkovNetwork)


@skipif_pgmpy_not_available
def test_one_repn_serves_several_queries():
    from conin.inference.variable_elimination.solve import _solve_repn

    repn = bex.cancer1_BN_constrained_oracle_conin().pgm.representation("factor")
    kw = dict(
        variables=None,
        show_progress=False,
        timing=False,
        write_uai_file=None,
        determine_variables=True,
    )
    r1 = _solve_repn(repn, evidence={"Smoker": 0}, **kw)
    r2 = _solve_repn(repn, evidence=None, **kw)
    assert "Smoker" not in r1.solution.states
    assert r2.solution.states["Smoker"] == 1
    assert repn.constraint_evidence == {("constraints", -1): 1}


@skipif_pgmpy_not_available
def test_write_uai_file_writes_the_constrained_model(tmp_path):
    from conin.io import load_model

    repn = bex.cancer1_BN_constrained_oracle_conin().pgm.representation("factor")
    filename = str(tmp_path / "model.uai")
    repn.create_model(write_uai_file=filename)
    loaded = load_model(filename)
    assert len(loaded.nodes) == len(repn.ve_model.nodes)


# ---------------------------------------------------------------------
# §9 regression tests (PR 4e)
# ---------------------------------------------------------------------


def _brute_force(pgm, constraint, evidence=None):
    """The constrained MAP assignment of a small MN, by enumeration."""
    import itertools

    from conin.common.log_potential import log_potential_conin

    evidence = evidence or {}
    nodes = list(pgm.nodes)
    best, best_score = None, None
    for values in itertools.product(*(pgm.states[n] for n in nodes)):
        a = dict(zip(nodes, values))
        if any(a[k] != v for k, v in evidence.items()) or not constraint.func(a):
            continue
        score = log_potential_conin(pgm, a)
        if best is None or score > best_score:
            best, best_score = a, score
    return best


@skipif_pgmpy_not_available
def test_constrained_mn_keeps_user_evidence():
    """§9: the ConstrainedMN overload replaced the user evidence with {}."""
    pgm = mex.ABC_conin().pgm
    con = _abc_all_different()
    cmodel = ConstrainedDiscreteMarkovNetwork(pgm, con)
    # The constrained optimum without evidence is A=0, B=2, C=1, so B=0 is
    # a real restriction (the old code ignored it and returned B=2).
    results = map_query(cmodel, method="variable_elimination", evidence={"B": 0})
    states = results.solution.states
    assert states["B"] == 0
    assert states == _brute_force(pgm, con, {"B": 0})


@skipif_pgmpy_not_available
def test_constrained_mn_does_not_force_a_scope_node_to_one():
    """§9: the MN factor oracle fixed the last scope node to 1 as evidence."""

    # With the old code, C (the last scope node) was fixed to 1, so a
    # constraint that requires C == 0 was infeasible.
    @oracle_constraint_fn(nodes=["A", "B", "C"])
    def c_is_zero(states):
        return states["C"] == 0

    pgm = mex.ABC_conin().pgm
    cmodel = ConstrainedDiscreteMarkovNetwork(pgm, c_is_zero)
    results = map_query(cmodel, method="variable_elimination")
    assert results.solution.states == _brute_force(pgm, c_is_zero)
    assert results.solution.states["C"] == 0
    assert cmodel.representation("factor").constraint_evidence == {}


@skipif_pgmpy_not_available
def test_constrained_mn_partial_query_has_only_the_query_variables():
    """§9 side effect: the forced scope node is no longer added to the result."""
    cmodel = mex.ABC_constrained_oracle_conin().pgm
    results = map_query(cmodel, method="variable_elimination", variables=["A", "B"])
    assert set(results.solution.states) == {"A", "B"}


@skipif_pgmpy_not_available
@pytest.mark.parametrize("example", [mex.ABC_conin, mex.ABC2_conin])
def test_constrained_mn_matches_brute_force(example):
    pgm = example().pgm
    con = _abc_all_different()
    results = map_query(
        ConstrainedDiscreteMarkovNetwork(pgm, con), method="variable_elimination"
    )
    assert results.solution.states == _brute_force(pgm, con)
