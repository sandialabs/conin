import pytest
import pyomo.opt

from conin.util import try_import

from conin.examples import bayesian_network as examples
from conin.inference import map_query

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination, BeliefPropagation

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None


# ===============================================================================
#
# pyomo DBDA_51 tests
#
# ===============================================================================

# TODO: replicate pgmpy tests with evidence here


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DBDA_51_pyomo():
    example = examples.DBDA_5_1_conin()
    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# toulbar2 DBDA_51 tests
#
# ===============================================================================

# TODO: replicate pgmpy tests with evidence here


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_DBDA_51_toulbar2():
    example = examples.DBDA_5_1_conin()
    results = map_query(example.pgm, method="toulbar2")
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# pgmpy BP tests
#
# ===============================================================================


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive():
    pgm = examples.DBDA_5_1_pgmpy().pgm
    infr1 = BeliefPropagation(pgm)

    evidence = {"test-result1": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1_pgmpy().pgm
    infr1 = BeliefPropagation(pgm)

    evidence = {"test-result1": 0, "test-result2": 1}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_two_positive():
    pgm = examples.DBDA_5_1_pgmpy().pgm
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
    pgm = examples.DBDA_5_1_pgmpy().pgm
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_variable_elimination_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1_pgmpy().pgm
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0, "test-result2": 1}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_varliable_elimination_two_positive():
    pgm = examples.DBDA_5_1_pgmpy().pgm
    infr1 = VariableElimination(pgm)

    evidence = {"test-result1": 0, "test-result2": 0}
    query_vars = ["disease-state"]
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {"disease-state": 1}
