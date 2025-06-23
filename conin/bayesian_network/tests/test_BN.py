import pytest
import pyomo.environ as pyo
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)
from . import test_cases

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
    pgm = test_cases.simple1_BN()
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
    pgm = test_cases.simple1_BN()
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
    pgm = test_cases.simple2_BN()
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
    pgm = test_cases.simple2_BN()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    model = create_BN_map_query_model(
        pgm=pgm, evidence={"A": 1}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    pgm = test_cases.cancer1_BN()
    q = {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"]
    )

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_Cancer():
    """
    Cancer model from pgmpy examples

    Evidence for Cancer
    """
    pgm = test_cases.cancer1_BN()
    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
        evidence={"Cancer": 0},
    )

    model = create_BN_map_query_model(
        pgm=pgm, evidence={"Cancer": 0}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_ALL_constrained1():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    pgm = test_cases.cancer1_BN()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    results = optimize_map_query_model(model, solver="glpk")  # num=1
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    cpgm = test_cases.cancer1_BN_constrained()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    pgm = test_cases.cancer2_BN()
    q = {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"]
    )

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_Cancer():
    """
    Cancer model from pgmpy examples

    Evidence for Cancer
    """
    pgm = test_cases.cancer2_BN()
    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
        evidence={"Cancer": 0},
    )

    model = create_BN_map_query_model(
        pgm=pgm, evidence={"Cancer": 0}
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_ALL_constrained1():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    pgm = test_cases.cancer2_BN()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    results = optimize_map_query_model(model, solver="glpk")  # num=1
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    cpgm = test_cases.cancer2_BN_constrained()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value
