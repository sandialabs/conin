import pytest
import pyomo.opt

from conin.util import try_import
from conin.inference.OptimizationInference import (
    IntegerProgrammingInference,
    DDBN_IntegerProgrammingInference,
)
import conin.markov_network.tests.examples
import conin.bayesian_network.tests.examples
import conin.hmm.tests.examples
import conin.dynamic_bayesian_network.tests.examples

with try_import() as pgmpy_available:
    import pgmpy

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


#
# DiscreteMarkovNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_ABC_conin():
    pgm = conin.markov_network.tests.examples.ABC_conin()
    inf = IntegerProgrammingInference(pgm)
    results = inf.map_query(solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_ABC_pgmpy():
    pgm = conin.markov_network.tests.examples.ABC_pgmpy()
    inf = IntegerProgrammingInference(pgm)
    results = inf.map_query(solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


#
# ConstrainedDiscreteMarkovNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_ABC_constrained():
    pgm = conin.markov_network.tests.examples.ABC_constrained_conin()
    inf = IntegerProgrammingInference(pgm)
    results = inf.map_query(solver=mip_solver)


#
# DiscreteBayesianNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_cancer1_ALL_conin():
    pgm = conin.bayesian_network.tests.examples.cancer1_BN_conin()
    inf = IntegerProgrammingInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
        solver=mip_solver,
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 1,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 0,
            "Pollution": 0,
            "Smoker": 0,
            "Xray": 0,
        }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 0,
            "Pollution": 0,
            "Xray": 0,
        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_cancer1_ALL_pgmpy():
    pgm = conin.bayesian_network.tests.examples.cancer1_BN_pgmpy()
    inf = IntegerProgrammingInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
        solver=mip_solver,
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 1,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 0,
            "Pollution": 0,
            "Smoker": 0,
            "Xray": 0,
        }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 0,
            "Pollution": 0,
            "Xray": 0,
        }


#
# ConstrainedBayesianNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_cancer1_constrained_conin():
    pgm = conin.bayesian_network.tests.examples.cancer1_BN_constrained_conin()
    inf = IntegerProgrammingInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
        solver=mip_solver,
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 0,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 1,
            "Pollution": 0,
            "Xray": 0,
        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_cancer1_constrained_pgmpy():
    pgm = conin.bayesian_network.tests.examples.cancer1_BN_constrained_pgmpy()
    inf = IntegerProgrammingInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"],
        solver=mip_solver,
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 0,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    with pytest.raises(RuntimeError):
        results = inf.map_query(
            variables=["Dyspnoea", "Pollution", "Xray"],
            evidence={"Cancer": 0},
            solver=mip_solver,
        )
        assert results.solution.variable_value == {
            "Dyspnoea": 1,
            "Pollution": 0,
            "Xray": 0,
        }


#
# HiddenMarkovModel tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_hmm1_test0():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = IntegerProgrammingInference(pgm)
    observed = ["o0", "o0", "o1", "o0", "o0"]
    results = inf.map_query(evidence=observed, solver=mip_solver)
    print(results)
    assert results.solution.variable_value == ["h0", "h0", "h0", "h0", "h0"]


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_hmm1_test1():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = IntegerProgrammingInference(pgm)
    observed = ["o0", "o1", "o1", "o1", "o1"]
    results = inf.map_query(evidence=observed, solver=mip_solver)
    assert results.solution.variable_value == ["h1", "h1", "h1", "h1", "h1"]


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_hmm1_test2():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = IntegerProgrammingInference(pgm)
    observed = {0: "o0", 1: "o0", 2: "o1", 3: "o0", 4: "o0"}
    results = inf.map_query(evidence=observed, solver=mip_solver)
    assert results.solution.variable_value == {
        0: "h0",
        1: "h0",
        2: "h0",
        3: "h0",
        4: "h0",
    }


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_hmm1_test3():
    pgm = conin.hmm.tests.examples.create_hmm1()
    inf = IntegerProgrammingInference(pgm)
    observed = {0: "o0", 1: "o1", 2: "o1", 3: "o1", 4: "o1"}
    results = inf.map_query(evidence=observed, solver=mip_solver)
    assert results.solution.variable_value == {
        0: "h1",
        1: "h1",
        2: "h1",
        3: "h1",
        4: "h1",
    }


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_chmm1_test0():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = IntegerProgrammingInference(pgm)
    observed = ["o0"] * 15
    results = inf.map_query(evidence=observed, solver=mip_solver)
    print(results)
    assert results.solution.variable_value == [
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


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_IntegerProgrammingInference_hmm1_test1():
    pgm = conin.hmm.tests.examples.create_chmm1_pyomo()
    inf = IntegerProgrammingInference(pgm)
    observed = ["o0"] + ["o1"] * 14
    results = inf.map_query(evidence=observed, solver=mip_solver)
    assert results.solution.variable_value == [
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
    ]


#
# DynamicBayesianNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DDBN_IntegerProgrammingInference_weather_conin():
    pgm = conin.dynamic_bayesian_network.tests.examples.weather_conin()
    inf = DDBN_IntegerProgrammingInference(pgm)

    results = inf.map_query(stop=4, solver=mip_solver)
    assert results.solution.variable_value == {
        ("H", 0): "Low",
        ("H", 1): "Low",
        ("H", 2): "Low",
        ("H", 3): "Low",
        ("H", 4): "Low",
        ("O", 0): "Dry",
        ("O", 1): "Dry",
        ("O", 2): "Dry",
        ("O", 3): "Dry",
        ("O", 4): "Dry",
        ("T", 0): "Hot",
        ("T", 1): "Hot",
        ("T", 2): "Hot",
        ("T", 3): "Hot",
        ("T", 4): "Hot",
        ("W", 0): "Sunny",
        ("W", 1): "Sunny",
        ("W", 2): "Sunny",
        ("W", 3): "Sunny",
        ("W", 4): "Sunny",
    }

    evidence = {
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

    with pytest.raises(RuntimeError):
        results = inf.map_query(stop=4, evidence=evidence, solver=mip_solver)
        # TODO - Confirm that this result makes sense
        assert results.solution.variable_value == {
            ("T", 0): "Hot",
            ("T", 1): "Hot",
            ("T", 2): "Hot",
            ("T", 3): "Hot",
            ("T", 4): "Hot",
            ("W", 0): "Cloudy",
            ("W", 1): "Cloudy",
            ("W", 2): "Cloudy",
            ("W", 3): "Cloudy",
            ("W", 4): "Cloudy",
        }


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DDBN_IntegerProgrammingInference_weather():
    pgm = conin.dynamic_bayesian_network.tests.examples.weather2_pgmpy()
    inf = DDBN_IntegerProgrammingInference(pgm)

    results = inf.map_query(stop=4, solver=mip_solver)
    assert results.solution.variable_value == {
        ("H", 0): "Low",
        ("H", 1): "Low",
        ("H", 2): "Low",
        ("H", 3): "Low",
        ("H", 4): "Low",
        ("O", 0): "Dry",
        ("O", 1): "Dry",
        ("O", 2): "Dry",
        ("O", 3): "Dry",
        ("O", 4): "Dry",
        ("T", 0): "Hot",
        ("T", 1): "Hot",
        ("T", 2): "Hot",
        ("T", 3): "Hot",
        ("T", 4): "Hot",
        ("W", 0): "Sunny",
        ("W", 1): "Sunny",
        ("W", 2): "Sunny",
        ("W", 3): "Sunny",
        ("W", 4): "Sunny",
    }

    evidence = {
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

    with pytest.raises(RuntimeError):
        results = inf.map_query(stop=4, evidence=evidence, solver=mip_solver)
        # TODO - Confirm that this result makes sense
        assert results.solution.variable_value == {
            ("T", 0): "Hot",
            ("T", 1): "Hot",
            ("T", 2): "Hot",
            ("T", 3): "Hot",
            ("T", 4): "Hot",
            ("W", 0): "Cloudy",
            ("W", 1): "Cloudy",
            ("W", 2): "Cloudy",
            ("W", 3): "Cloudy",
            ("W", 4): "Cloudy",
        }


#
# ConstrainedDynamicBayesianNetwork tests
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DDBN_IntegerProgrammingInference_weather_constrained_conin():
    pgm = conin.dynamic_bayesian_network.tests.examples.weather_constrained_conin()
    inf = DDBN_IntegerProgrammingInference(pgm)

    results = inf.map_query(stop=4, solver=mip_solver)
    assert results.solution.variable_value == {
        ("H", 0): "Low",
        ("H", 1): "Low",
        ("H", 2): "Low",
        ("H", 3): "High",
        ("H", 4): "High",
        ("O", 0): "Dry",
        ("O", 1): "Dry",
        ("O", 2): "Dry",
        ("O", 3): "Wet",
        ("O", 4): "Wet",
        ("T", 0): "Hot",
        ("T", 1): "Hot",
        ("T", 2): "Hot",
        ("T", 3): "Hot",
        ("T", 4): "Mild",
        ("W", 0): "Sunny",
        ("W", 1): "Sunny",
        ("W", 2): "Sunny",
        ("W", 3): "Rainy",
        ("W", 4): "Rainy",
    }

    evidence = {
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

    with pytest.raises(RuntimeError):
        results = inf.map_query(stop=4, evidence=evidence, solver=mip_solver)
        assert results.solution.variable_value == {
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


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DDBN_IntegerProgrammingInference_weather_constrained_pgmpy():
    pgm = conin.dynamic_bayesian_network.tests.examples.weather_constrained_pgmpy()
    inf = DDBN_IntegerProgrammingInference(pgm)

    results = inf.map_query(stop=4, solver=mip_solver)
    assert results.solution.variable_value == {
        ("H", 0): "Low",
        ("H", 1): "Low",
        ("H", 2): "Low",
        ("H", 3): "High",
        ("H", 4): "High",
        ("O", 0): "Dry",
        ("O", 1): "Dry",
        ("O", 2): "Dry",
        ("O", 3): "Wet",
        ("O", 4): "Wet",
        ("T", 0): "Hot",
        ("T", 1): "Hot",
        ("T", 2): "Hot",
        ("T", 3): "Hot",
        ("T", 4): "Mild",
        ("W", 0): "Sunny",
        ("W", 1): "Sunny",
        ("W", 2): "Sunny",
        ("W", 3): "Rainy",
        ("W", 4): "Rainy",
    }

    evidence = {
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

    with pytest.raises(RuntimeError):
        results = inf.map_query(stop=4, evidence=evidence, solver=mip_solver)
        assert results.solution.variable_value == {
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
