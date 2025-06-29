import pytest
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)
from . import examples

try:
    from pgmpy.inference import VariableElimination, BeliefPropagation

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive():
    pgm = examples.DBDA_5_1()
    infr1 = BeliefPropagation(pgm)

    evidence = {'test-result1': 0}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1()
    infr1 = BeliefPropagation(pgm)

    evidence = {'test-result1': 0, 'test-result2': 1}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_belief_propagation_two_positive():
    pgm = examples.DBDA_5_1()
    infr1 = BeliefPropagation(pgm)

    evidence = {'test-result1': 0, 'test-result2': 0}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_variable_elimination_one_positive():
    pgm = examples.DBDA_5_1()
    infr1 = VariableElimination(pgm)

    evidence = {'test-result1': 0}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_variable_elimination_one_positive_and_one_negative():
    pgm = examples.DBDA_5_1()
    infr1 = VariableElimination(pgm)

    evidence = {'test-result1': 0, 'test-result2': 1}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_DBDA_51_varliable_elimination_two_positive():
    pgm = examples.DBDA_5_1()
    infr1 = VariableElimination(pgm)

    evidence = {'test-result1': 0, 'test-result2': 0}
    query_vars = ['disease-state']
    results = infr1.map_query(variables=query_vars, evidence=evidence)

    assert results == {'disease-state':1}


