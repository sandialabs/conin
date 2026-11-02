"""Tests for the "variable_elimination" representations and variable elimination."""

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
from conin.algorithms.inference import map_query
from conin.algorithms.inference.variable_elimination import (
    solve_variable_elimination_map_query_model,
)
from conin.representations import supported_representations
from conin.representations.variable_elimination import (
    VariableEliminationBNRepresentation,
    VariableEliminationDBNRepresentation,
    VariableEliminationHMMRepresentation,
    VariableEliminationMNRepresentation,
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
        assert ("variable_elimination", family) in pairs
    for family in ("pyomo", "toulbar2", "oracle", "mvr"):
        assert ("variable_elimination", family) not in pairs


def test_factor_rejects_pyomo_constraints():
    @pyomo_constraint_fn()
    def noop(M, D):
        pass

    cmodel = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm, noop)
    with pytest.raises(
        InvalidInputError, match="No 'variable_elimination' representation"
    ):
        cmodel.representation("variable_elimination")


def test_variable_elimination_rejects_pyomo_constraints():
    """Regression: this used to raise a TypeError."""
    example = bex.cancer1_BN_constrained_pyomo_conin()
    with pytest.raises(
        InvalidInputError, match="No 'variable_elimination' representation"
    ):
        map_query(example.pgm, method="variable_elimination")


@pytest.mark.parametrize(
    "cmodel,params,cls",
    [
        (
            ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm),
            {},
            VariableEliminationMNRepresentation,
        ),
        (
            ConstrainedDiscreteBayesianNetwork(bex.cancer1_BN_conin().pgm),
            {},
            VariableEliminationBNRepresentation,
        ),
        (
            ConstrainedDynamicDiscreteBayesianNetwork(dex.weather_conin().pgm),
            dict(stop=4),
            VariableEliminationDBNRepresentation,
        ),
        (
            ConstrainedHiddenMarkovModel(hex_.create_hmm1()),
            dict(stop=4),
            VariableEliminationHMMRepresentation,
        ),
    ],
)
def test_factor_representation_classes(cmodel, params, cls):
    repn = cmodel.representation("variable_elimination", **params)
    assert type(repn) is cls
    assert repn.kind == "variable_elimination"
    # Unconstrained: the VE model has the factors (or CPDs) of the model.
    assert isinstance(repn.ve_model, DiscreteMarkovNetwork)
    assert set(repn.ve_model.nodes) == set(repn.nodes)
    expected = repn.pgm.factors if hasattr(repn.pgm, "factors") else repn.pgm.cpds
    assert len(repn.ve_model.factors) == len(expected)
    if isinstance(repn.pgm, DiscreteMarkovNetwork):
        assert repn.ve_model is repn.pgm


# ---------------------------------------------------------------------
# Structure params
# ---------------------------------------------------------------------


def test_dbn_structure_params():
    dbn = dex.weather_conin().pgm
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork(dbn)
    default = cmodel.representation("variable_elimination")
    assert (default.start, default.stop) == (0, 1)
    # (start > 0 fails in create_bn_from_dbn for every backend.)
    repn = cmodel.representation("variable_elimination", start=0, stop=3)
    assert (repn.start, repn.stop) == (0, 3)
    assert repn.constraint_data() == munch.Munch(start=0, stop=3, T=[0, 1, 2, 3])
    assert set(repn.nodes) == set(create_bn_from_dbn(dbn=dbn, start=0, stop=3).nodes)


def test_hmm_structure_params():
    hmm = hex_.create_hmm1()
    cmodel = ConstrainedHiddenMarkovModel(hmm)
    with pytest.raises(InvalidInputError, match="needs a stop time"):
        cmodel.representation("variable_elimination")
    repn = cmodel.representation("variable_elimination", stop=3)
    assert (repn.start, repn.stop) == (0, 3)
    assert repn.constraint_data() == munch.Munch(
        start=0,
        stop=3,
        hmm=munch.Munch(T=[0, 1, 2, 3], hidden_to_internal=hmm.hidden_to_internal),
    )
    unrolled = create_bn_from_dbn(dbn=create_dbn_from_hmm(hmm), start=0, stop=3)
    assert set(repn.nodes) == set(unrolled.nodes)


# ---------------------------------------------------------------------
# The constraint data each model passes
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
    ).representation("variable_elimination")
    ConstrainedDiscreteBayesianNetwork(
        bex.cancer1_BN_conin().pgm, _recording_constraint(seen, ["Smoker"])
    ).representation("variable_elimination")
    assert seen == [None, None]


def test_constraint_data_dbn_and_hmm():
    seen = []
    ConstrainedDynamicDiscreteBayesianNetwork(
        dex.simple1_DDBN_conin().pgm, _recording_constraint(seen, [("A", 0)])
    ).representation("variable_elimination", stop=2)
    hmm = hex_.create_hmm1()
    ConstrainedHiddenMarkovModel(
        hmm, _recording_constraint(seen, [("H", 0)])
    ).representation("variable_elimination", stop=2)
    assert seen == [
        munch.Munch(start=0, stop=2, T=[0, 1, 2]),
        munch.Munch(
            start=0,
            stop=2,
            hmm=munch.Munch(T=[0, 1, 2], hidden_to_internal=hmm.hidden_to_internal),
        ),
    ]


def test_constraints_are_materialised_at_build_time():
    seen = []
    cmodel = ConstrainedDiscreteBayesianNetwork(
        bex.cancer1_BN_conin().pgm, _recording_constraint(seen, ["Smoker"])
    )
    repn = cmodel.representation("variable_elimination")
    assert len(seen) == 1
    # One indicator factor over the scope, no new node
    assert repn.ve_model.factors[-1].nodes == ["Smoker"]
    assert set(repn.ve_model.nodes) == set(repn.nodes)
    solve_variable_elimination_map_query_model(repn.ve_model, evidence={"Smoker": 0})
    assert len(seen) == 1  # queries don't re-materialise


# ---------------------------------------------------------------------
# Queries don't mutate the representation
# ---------------------------------------------------------------------


def test_one_repn_serves_several_queries():
    repn = bex.cancer1_BN_constrained_oracle_conin().pgm.representation(
        "variable_elimination"
    )
    n_factors = len(repn.ve_model.factors)
    r1 = solve_variable_elimination_map_query_model(
        repn.ve_model, evidence={"Smoker": 0}
    )
    r2 = solve_variable_elimination_map_query_model(repn.ve_model)
    assert "Smoker" not in r1.solution.states
    assert r2.solution.states["Smoker"] == 1
    assert len(repn.ve_model.factors) == n_factors


def test_write_uai_file_writes_the_constrained_model(tmp_path):
    from conin.io import load_model

    cmodel = bex.cancer1_BN_constrained_oracle_conin().pgm
    repn = cmodel.representation("variable_elimination")
    filename = str(tmp_path / "model.uai")
    map_query(cmodel, method="variable_elimination", write_uai_file=filename)
    loaded = load_model(filename)
    assert isinstance(loaded, DiscreteMarkovNetwork)
    assert len(loaded.nodes) == len(repn.ve_model.nodes)
    assert len(loaded.factors) == len(repn.ve_model.factors)


# ---------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------


def _brute_force(pgm, constraint, evidence=None):
    """The constrained MAP assignment of a small MN, by enumeration."""
    import itertools

    from conin.algorithms.scoring import log_potential

    evidence = evidence or {}
    nodes = list(pgm.nodes)
    best, best_score = None, None
    for values in itertools.product(*(pgm.states[n] for n in nodes)):
        a = dict(zip(nodes, values))
        if any(a[k] != v for k, v in evidence.items()) or not constraint.func(a):
            continue
        score = log_potential(pgm, a)
        if best is None or score > best_score:
            best, best_score = a, score
    return best


def test_constrained_mn_keeps_user_evidence():
    """Regression: the ConstrainedMN overload replaced the user evidence with {}."""
    pgm = mex.ABC_conin().pgm
    con = _abc_all_different()
    cmodel = ConstrainedDiscreteMarkovNetwork(pgm, con)
    # The constrained optimum without evidence is A=0, B=2, C=1, so B=0 is
    # a real restriction (the old code ignored it and returned B=2).
    results = map_query(cmodel, method="variable_elimination", evidence={"B": 0})
    states = results.solution.states
    assert states["B"] == 0
    assert states == _brute_force(pgm, con, {"B": 0})


def test_constrained_mn_does_not_force_a_scope_node_to_one():
    """Regression: the MN factor oracle fixed the last scope node to 1 as evidence."""

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


def test_constrained_mn_partial_query_has_only_the_query_variables():
    """The result has only the query variables (no forced scope node)."""
    cmodel = mex.ABC_constrained_oracle_conin().pgm
    results = map_query(cmodel, method="variable_elimination", variables=["A", "B"])
    assert set(results.solution.states) == {"A", "B"}


@pytest.mark.parametrize("example", [mex.ABC_conin, mex.ABC2_conin])
def test_constrained_mn_matches_brute_force(example):
    pgm = example().pgm
    con = _abc_all_different()
    results = map_query(
        ConstrainedDiscreteMarkovNetwork(pgm, con), method="variable_elimination"
    )
    assert results.solution.states == _brute_force(pgm, con)
