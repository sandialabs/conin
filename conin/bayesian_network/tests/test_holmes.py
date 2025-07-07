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
def test_holmes1():
    pgm = examples.holmes()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"W": "w", "G": "g"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes2():
    pgm = examples.holmes()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"A": "-a"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes3():
    pgm = examples.holmes()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"A": "a"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes4():
    pgm = examples.holmes()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"B": "b"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_holmes5():
    pgm = examples.holmes()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"R": "r", "W": "-w"}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    model = create_BN_map_query_model(pgm=pgm, variables=variables, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
