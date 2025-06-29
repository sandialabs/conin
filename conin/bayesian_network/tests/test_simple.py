import pytest
import pyomo.environ as pyo
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)
from . import examples

try:
    from pgmpy.inference import VariableElimination

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B
    """
    pgm = examples.simple1_BN()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple1_BN()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    model = create_BN_map_query_model(
        pgm=pgm, evidence={"A": 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_ALL():
    """
    A -> B
    """
    pgm = examples.simple2_BN()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_B():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple2_BN()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    model = create_BN_map_query_model(
        pgm=pgm, evidence={"A": 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
