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
