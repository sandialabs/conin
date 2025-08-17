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
def test_holmes1():
    pgm = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"W": "w", "G": "g"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, variables=variables, evidence=evidence
        )
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes2():
    pgm = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"A": "-a"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, variables=variables, evidence=evidence
        )
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes3():
    pgm = examples.holmes_pgmpy()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"A": "a"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, variables=variables, evidence=evidence
        )
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes4():
    pgm = examples.holmes_pgmpy()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"B": "b"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, variables=variables, evidence=evidence
        )
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes5():
    pgm = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"R": "r", "W": "-w"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, variables=variables, evidence=evidence
        )
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value
