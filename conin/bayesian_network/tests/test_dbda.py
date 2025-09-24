import pytest
import pyomo.opt

from conin.util import try_import
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)

from . import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination, BeliefPropagation

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


#
# conin DBDA_51 tests
#
# TODO: replicate pgmpy tests with evidence here
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DBDA_51_conin():
    pgm = examples.DBDA_5_1_conin()
    q = {"disease-state": 1, "test-result1": 1, "test-result2": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver=mip_solver)
    assert q == results.solution.variable_value


#
# pgmpy BP tests
#


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = BeliefPropagation(pgm)

    evidence = {"test-result1": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = BeliefPropagation(pgm)

    evidence = {"test-result1": 0, "test-result2": 1}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_two_positive():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = BeliefPropagation(pgm)

    evidence = {"test-result1": 0, "test-result2": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


#
# pgmpy VE tests
#


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_variable_elimination_one_positive():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_variable_elimination_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0, "test-result2": 1}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_varliable_elimination_two_positive():
    pgm = examples.DBDA_5_1_pgmpy()
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0, "test-result2": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}
