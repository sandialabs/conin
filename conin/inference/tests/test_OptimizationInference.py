import os
import pyomo.opt

import conin.markov_network.examples
import conin.bayesian_network.examples
import conin.dynamic_bayesian_network.examples
import conin.hidden_markov_model.tests.examples

from conin.inference.map_query import map_query
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

import pytest

skipif_no_mip_solver = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)
ip_formulations = pytest.mark.parametrize(
    "ip_formulation", [None, "markov_network", "network_flow"]
)

cwd = os.path.dirname(__file__)
testfile_lp = os.path.join(cwd, "test.lp")

#
# DiscreteMarkovNetwork tests
#


@skipif_no_mip_solver
def test_IntegerProgrammingInference_ABC_conin():
    example = conin.markov_network.examples.ABC_conin()
    results = map_query(example.pgm, inference="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(
        example.pgm,
        inference="integer_program",
        solver=mip_solver,
        write_lp_file=testfile_lp,
    )
    assert os.path.exists(testfile_lp)
    os.remove(testfile_lp)


#
# ConstrainedDiscreteMarkovNetwork tests
#

#
# DiscreteBayesianNetwork tests
#

#
#    # TODO - Confirm that these marginalized results are correct
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            solver=mip_solver,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }

#
#    # TODO - Confirm that these marginalized results are correct
#
#    with pytest.raises(RuntimeError):
#        results = inf.map_query(
#            variables=["Dyspnoea", "Pollution", "Xray"],
#            evidence={"Cancer": 0},
#            solver=mip_solver,
#        )
#        assert results.solution.states == {
#            "Dyspnoea": 0,
#            "Pollution": 0,
#            "Xray": 0,
#        }

#
# ConstrainedBayesianNetwork tests
#

#
# HiddenMarkovModel tests
#


@skipif_no_mip_solver
@ip_formulations
def test0_IntegerProgrammingInference_hmm1(ip_formulation):
    pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = map_query(
        pgm,
        inference="integer_program",
        evidence=observed,
        solver=mip_solver,
        ip_formulation=ip_formulation,
    )
    assert results.solution.states == ["h0", "h0", "h0", "h0", "h0"]
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(
        pgm,
        inference="integer_program",
        evidence=observed,
        solver=mip_solver,
        write_lp_file=testfile_lp,
    )
    assert os.path.exists(testfile_lp)
    os.remove(testfile_lp)


@skipif_no_mip_solver
@ip_formulations
def test0_IntegerProgrammingInference_chmm1(ip_formulation):
    pgm = conin.hidden_markov_model.tests.examples.create_chmm1_pyomo()
    observed = ["o0"] * 15
    results = map_query(
        pgm,
        inference="integer_program",
        evidence=observed,
        solver=mip_solver,
        ip_formulation=ip_formulation,
    )
    assert results.solution.states == [
        "h1",
        "h1",
        "h1",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
        "h0",
    ]
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    results = map_query(
        pgm,
        inference="integer_program",
        evidence=observed,
        solver=mip_solver,
        write_lp_file=testfile_lp,
    )
    assert os.path.exists(testfile_lp)
    os.remove(testfile_lp)


#
# DynamicBayesianNetwork tests
#

weather_evidence = {
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
}

q_unconstrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Mild",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Cloudy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}

q_constrained = {
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Mild",
    ("T", 2): "Cold",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Rainy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


@skipif_no_mip_solver
def test_DPGM_IntegerProgrammingInference_weather_conin():
    example = conin.dynamic_bayesian_network.examples.weather_conin()

    # without evidence
    results = map_query(
        example.pgm, inference="integer_program", stop=4, solver=mip_solver
    )
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # with evidence
    results = map_query(
        example.pgm,
        inference="integer_program",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_unconstrained == results.solution.states

    results = map_query(
        example.pgm,
        inference="integer_program",
        stop=4,
        solver=mip_solver,
        write_lp_file=testfile_lp,
    )
    assert os.path.exists(testfile_lp)
    os.remove(testfile_lp)


#
# ConstrainedDynamicBayesianNetwork tests
#

#
# Tests for map_query() dispatch function
#


@skipif_no_mip_solver
@pytest.mark.parametrize(
    "example_factory",
    [
        conin.markov_network.examples.ABC_conin,
        conin.markov_network.examples.ABC_constrained_pyomo_conin,
        conin.bayesian_network.examples.cancer1_BN_conin,
        conin.bayesian_network.examples.cancer1_BN_constrained_pyomo_conin,
    ],
)
def test_map_query_static_models(example_factory):
    """Test map_query dispatch for static models (MN and BN)."""
    example = example_factory()
    results = map_query(example.pgm, inference="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float


@skipif_pgmpy_not_available
@skipif_no_mip_solver
@pytest.mark.parametrize(
    "example_factory",
    [
        conin.markov_network.examples.ABC_pgmpy,
        conin.bayesian_network.examples.cancer1_BN_pgmpy,
    ],
)
def test_map_query_static_models_pgmpy(example_factory):
    """Test map_query dispatch for pgmpy static models."""
    example = example_factory()
    results = map_query(example.pgm, inference="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


@skipif_no_mip_solver
@ip_formulations
@pytest.mark.parametrize(
    "pgm_factory,evidence,expected",
    [
        (
            lambda: conin.hidden_markov_model.tests.examples.create_hmm1(),
            ["o0", "o0", "o1", "o0", "o0"],
            ["h0", "h0", "h0", "h0", "h0"],
        ),
        (
            lambda: conin.hidden_markov_model.tests.examples.create_hmm1(),
            ["o0", "o1", "o1", "o1", "o1"],
            ["h1", "h1", "h1", "h1", "h1"],
        ),
        (
            lambda: conin.hidden_markov_model.tests.examples.create_hmm1(),
            {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"},
            {0: "h0", 1: "h0", 2: "h0", 3: "h0", 4: "h0"},
        ),
    ],
)
def test_map_query_hmm(pgm_factory, evidence, expected, ip_formulation):
    """Test map_query dispatch for HMM models."""
    pgm = pgm_factory()
    results = map_query(
        pgm,
        inference="integer_program",
        evidence=evidence,
        solver=mip_solver,
        ip_formulation=ip_formulation,
    )
    assert results.solution.states == expected
    assert hasattr(results, "solvetime") and type(results.solvetime) is float


@skipif_no_mip_solver
@ip_formulations
@pytest.mark.parametrize(
    "pgm_factory,evidence,expected",
    [
        (
            lambda: conin.hidden_markov_model.tests.examples.create_chmm1_pyomo(),
            ["o0"] * 15,
            [
                "h1",
                "h1",
                "h1",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
            ],
        ),
        (
            lambda: conin.hidden_markov_model.tests.examples.create_chmm1_pyomo(),
            ["o0"] + ["o1"] * 14,
            [
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h0",
                "h1",
                "h1",
                "h1",
                "h1",
                "h1",
            ],
        ),
    ],
)
def test_map_query_chmm(pgm_factory, evidence, expected, ip_formulation):
    """Test map_query dispatch for constrained HMM models."""
    pgm = pgm_factory()
    results = map_query(
        pgm,
        inference="integer_program",
        evidence=evidence,
        solver=mip_solver,
        ip_formulation=ip_formulation,
    )
    assert results.solution.states == expected
    assert hasattr(results, "solvetime") and type(results.solvetime) is float


@skipif_no_mip_solver
def test_map_query_dbn():
    """Test map_query dispatch for dynamic Bayesian network."""
    example = conin.dynamic_bayesian_network.examples.weather_conin()

    # without evidence
    results = map_query(
        example.pgm, inference="integer_program", stop=4, solver=mip_solver
    )
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # with evidence
    results = map_query(
        example.pgm,
        inference="integer_program",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_unconstrained == results.solution.states


@skipif_no_mip_solver
def test_map_query_constrained_dbn():
    """Test map_query dispatch for constrained dynamic Bayesian network."""
    example = conin.dynamic_bayesian_network.examples.weather_constrained_pyomo_conin()

    # without evidence
    results = map_query(
        example.pgm, inference="integer_program", stop=4, solver=mip_solver
    )
    assert results.solution.states == example.solutions[0].states
    assert hasattr(results, "solvetime") and type(results.solvetime) is float

    # with evidence
    results = map_query(
        example.pgm,
        inference="integer_program",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_constrained == results.solution.states


@skipif_pgmpy_not_available
@skipif_no_mip_solver
def test_map_query_dbn_pgmpy():
    """Test map_query dispatch for pgmpy dynamic Bayesian network."""
    example = conin.dynamic_bayesian_network.examples.weather2_pgmpy()

    # without evidence
    results = map_query(
        example.pgm, inference="integer_program", stop=4, solver=mip_solver
    )
    assert results.solution.states == example.solutions[0].states

    # with evidence
    results = map_query(
        example.pgm,
        inference="integer_program",
        stop=4,
        evidence=weather_evidence,
        solution_with_evidence=True,
    )
    assert q_unconstrained == results.solution.states


def test_map_query_unsupported_type():
    """Test that map_query raises TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported model type"):
        map_query("not_a_model", inference="integer_program")
