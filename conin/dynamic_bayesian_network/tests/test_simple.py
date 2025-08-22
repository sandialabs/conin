import pytest

from conin.util import try_import
from conin.dynamic_bayesian_network import (
    create_DDBN_map_query_model,
    optimize_map_query_model,
)

from . import examples

with try_import() as pgmpy_available:
    import pgmpy


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_ALL():
    """
    Z
    """
    G = examples.simple0_DDBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_ALL():
    """
    Z
    """
    G = examples.simple2_DDBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_ALL():
    """
    Z
    """
    G = examples.simple3_DDBN()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B

    No evidence
    """
    G = examples.simple1_DDBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = examples.simple1_DDBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DDBN_map_query_model(
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
    cpgm = examples.simple1_DDBN_constrained()
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
    G = examples.simple3_DDBN()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple3_B():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = examples.simple3_DDBN()
    q = {
        ("A", 1): 0,
        ("B", 0): 0,
        ("B", 1): 1,
    }

    model = create_DDBN_map_query_model(
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
    cpgm = examples.simple3_DDBN_constrained()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value
