import pytest
from . import test_cases

try:
    import pgmpy

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False

from conin.dynamic_bayesian_network import (
    create_DBN_map_query_model,
    optimize_map_query_model,
)


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_ALL():
    """
    Z
    """
    G = test_cases.simple0_DBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_ALL():
    """
    Z
    """
    G = test_cases.simple2_DBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL():
    """
    Z
    """
    G = test_cases.simple3_DBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B

    No evidence
    """
    G = test_cases.simple1_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = test_cases.simple1_DBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(
        pgm=G, evidence={("A", 0): 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL_constrained():
    """
    A -> B

    No evidence
    """
    cpgm = test_cases.simple1_DBN_constrained()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL():
    """
    A -> B

    No evidence
    """
    G = test_cases.simple3_DBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = test_cases.simple3_DBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DBN_map_query_model(
        pgm=G, evidence={("A", 0): 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL_constrained():
    """
    A -> B

    No evidence
    """
    cpgm = test_cases.simple3_DBN_constrained()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather1_A():
    """
    Test with TabularCPD representation
    """
    pgm = test_cases.pgmpy_weather1()
    q = {
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

    model = create_DBN_map_query_model(pgm=pgm, stop=4)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather1_B():
    """
    Test with TabularCPD representation
    """
    pgm = test_cases.pgmpy_weather1()
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
    q = {
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

    model = create_DBN_map_query_model(pgm=pgm, stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather2_A():
    """
    Test with TabularCPD representation
    """
    pgm = test_cases.pgmpy_weather2()
    q = {
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

    model = create_DBN_map_query_model(pgm=pgm, stop=4)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather2_B():
    """
    Test with TabularCPD representation
    """
    pgm = test_cases.pgmpy_weather2()
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
    q = {
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

    model = create_DBN_map_query_model(pgm=pgm, stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather2_B_constrained1():
    """
    Test with TabularCPD representation
    """
    cpgm = test_cases.pgmpy_weather_constrained1()
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
    q = {
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

    model = cpgm.create_map_query_model(stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
