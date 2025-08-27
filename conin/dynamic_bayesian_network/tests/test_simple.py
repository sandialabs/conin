import pytest

from conin.util import try_import
from conin.dynamic_bayesian_network import (
    create_DDBN_map_query_model,
    optimize_map_query_model,
)

from . import examples

with try_import() as pgmpy_available:
    import pgmpy


#
# simple0
#


def test_simple0_ALL_conin():
    """
    Z
    """
    G = examples.simple0_DDBN_conin()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_DDBN1_ALL_pgmpy():
    """
    Z
    """
    G = examples.simple0_DDBN1_pgmpy()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple0_DDBN2_ALL_pgmpy():
    """
    Z
    """
    G = examples.simple0_DDBN2_pgmpy()
    q = {("Z", 0): 1, ("Z", 1): 0}

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


#
# simple1
#


def test_simple1_ALL_conin():
    """
    A -> B

    No evidence
    """
    G = examples.simple1_DDBN_conin()
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
def test_simple1_ALL_pgmpy():
    """
    A -> B

    No evidence
    """
    G = examples.simple1_DDBN_pgmpy()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_simple1_B_conin():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = examples.simple1_DDBN_conin()
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
def test_simple1_B_pgmpy():
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


def test_simple1_ALL_constrained_conin():
    """
    A -> B

    No evidence
    """
    cpgm = examples.simple1_DDBN_constrained_conin()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple1_ALL_constrained_pgmpy():
    """
    A -> B

    No evidence
    """
    cpgm = examples.simple1_DDBN_constrained_pgmpy()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


#
# simple2
#


def test_simple2_ALL_conin():
    """
    A -> B

    No evidence
    """
    G = examples.simple2_DDBN_conin()
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
def test_simple2_ALL_pgmpy():
    """
    A -> B

    No evidence
    """
    G = examples.simple2_DDBN_pgmpy()
    q = {
        ("A", 0): 0,
        ("A", 1): 1,
        ("B", 0): 1,
        ("B", 1): 0,
    }

    model = create_DDBN_map_query_model(pgm=G)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_simple2_B_conin():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = examples.simple2_DDBN_conin()
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
def test_simple2_B_pgmpy():
    """
    A -> B

    Evidence: A_0 = 1
    """
    G = examples.simple2_DDBN_pgmpy()
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


def test_simple2_ALL_constrained_conin():
    """
    A -> B

    No evidence
    """
    cpgm = examples.simple2_DDBN_constrained_conin()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_simple2_ALL_constrained_pgmpy():
    """
    A -> B

    No evidence
    """
    cpgm = examples.simple2_DDBN_constrained_pgmpy()
    q = {
        ("A", 0): 0,
        ("A", 1): 0,
        ("B", 0): 1,
        ("B", 1): 1,
    }

    results = optimize_map_query_model(cpgm.create_map_query_model(), solver="glpk")
    assert q == results.solution.variable_value
