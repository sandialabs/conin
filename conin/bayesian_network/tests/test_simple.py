import pytest
import pyomo.environ as pyo

from conin.util import try_import
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)
from conin.bayesian_network.model import convert_to_DiscreteBayesianNetwork

from . import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL():
    """
    A -> B
    """
    pgm = examples.simple1_BN_pgmpy()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_B():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple1_BN_pgmpy()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
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
    pgm = examples.simple2_BN_pgmpy()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_B():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple2_BN_pgmpy()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"A": 1}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value
