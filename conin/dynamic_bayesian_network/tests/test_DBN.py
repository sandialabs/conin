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
