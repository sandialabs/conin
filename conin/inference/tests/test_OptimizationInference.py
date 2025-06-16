import pytest

from conin.inference.OptimizationInference import OptimizationInference
import conin.markov_network.tests.test_cases
import conin.bayesian_network.tests.test_cases
import conin.dynamic_bayesian_network.tests.test_cases

#
# MarkovNetwork tests
#


def test_OptimizationInference_ABC():
    pgm = conin.markov_network.tests.test_cases.ABC()
    inf = OptimizationInference(pgm)
    results = inf.map_query(solver="glpk")
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


#
# ConstrainedMarkovNetwork tests
#


def test_OptimizationInference_ABC_constrained():
    pgm = conin.markov_network.tests.test_cases.ABC_constrained()
    inf = OptimizationInference(pgm)
    results = inf.map_query(solver="glpk")


#
# BayesianNetwork tests
#


def test_OptimizationInference_cancer2_ALL():
    pgm = conin.bayesian_network.tests.test_cases.cancer2_BN()
    inf = OptimizationInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"], solver="glpk"
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 1,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    results = inf.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
        evidence={"Cancer": 0},
        solver="glpk",
    )
    assert results.solution.variable_value == {
        "Dyspnoea": 0,
        "Pollution": 0,
        "Smoker": 0,
        "Xray": 0,
    }

    # TODO - Confirm that these marginalized results are correct

    results = inf.map_query(
        variables=["Dyspnoea", "Pollution", "Xray"],
        evidence={"Cancer": 0},
        solver="glpk",
    )
    assert results.solution.variable_value == {
        "Dyspnoea": 0,
        "Pollution": 0,
        "Smoker": 0,
        "Xray": 0,
    }

#
# ConstrainedBayesianNetwork tests
#

def test_OptimizationInference_cancer2_constrained():
    pgm = conin.bayesian_network.tests.test_cases.cancer2_BN_constrained()
    inf = OptimizationInference(pgm)

    results = inf.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"], solver="glpk"
    )
    assert results.solution.variable_value == {
        "Cancer": 1,
        "Dyspnoea": 0,
        "Pollution": 0,
        "Smoker": 1,
        "Xray": 1,
    }

    # TODO - Confirm that these marginalized results are correct

    results = inf.map_query(
        variables=["Dyspnoea", "Pollution", "Xray"],
        evidence={"Cancer": 0},
        solver="glpk",
    )
    assert results.solution.variable_value == {
        "Dyspnoea": 1,
        "Pollution": 0,
        "Smoker": 0,
        "Xray": 0,
    }

#
# DynamicBayesianNetwork tests
#

def test_OptimizationInference_weather():
    pgm = conin.dynamic_bayesian_network.tests.test_cases.pgmpy_weather2()
    inf = OptimizationInference(pgm)

    results = inf.map_query(solver='glpk')
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
    results = inf.map_query(evidence=evidence, solver='glpk')
    assert results.solution.variable_value == {
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

#
# ConstrainedDynamicBayesianNetwork tests
#
def test_OptimizationInference_weather_constrained():
    pgm = conin.dynamic_bayesian_network.tests.test_cases.pgmpy_weather_constrained1()
    inf = OptimizationInference(pgm)

    results = inf.map_query(solver='glpk')
    assert results.solution.variable_value == {
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
    results = inf.map_query(evidence=evidence, solver='glpk')
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

