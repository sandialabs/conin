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


def test_cancer1_conin_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    pgm = examples.cancer1_BN_conin()
    q = {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_pgmpy_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    pgm = examples.cancer1_BN_pgmpy()
    q = {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"]
    )

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_cancer1_conin_Cancer():
    """
    Cancer model from pgmpy examples

    Evidence for Cancer
    """
    pgm = examples.cancer1_BN_conin()
    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"Cancer": 0}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer1_pgmpy_Cancer():
    """
    Cancer model from pgmpy examples

    Evidence for Cancer
    """
    pgm = examples.cancer1_BN_pgmpy()
    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
        evidence={"Cancer": 0},
    )

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"Cancer": 0}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


def test_cancer1_conin_ALL_constrained1():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    pgm = examples.cancer1_BN_conin()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    results = optimize_map_query_model(model, solver="glpk")  # num=1
    assert q == results.solution.variable_value


def test_cancer1_conin_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    cpgm = examples.cancer1_BN_constrained_conin()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_pgmpy_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    pgm = examples.cancer2_BN_pgmpy()
    q = {"Cancer": 1, "Dyspnoea": 1, "Pollution": 0, "Smoker": 1, "Xray": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"]
    )

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_pgmpy_Cancer():
    """
    Cancer model from pgmpy examples

    Evidence for Cancer
    """
    pgm = examples.cancer2_BN_pgmpy()
    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(
        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
        evidence={"Cancer": 0},
    )

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"Cancer": 0}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver="glpk")
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_pgmpy_ALL_constrained1():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    pgm = examples.cancer2_BN_pgmpy()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    pgm = convert_to_DiscreteBayesianNetwork(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    model.c = pyo.ConstraintList()
    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)

    results = optimize_map_query_model(model, solver="glpk")  # num=1
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_cancer2_pgmpy_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    cpgm = examples.cancer2_BN_constrained_pgmpy()
    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value
