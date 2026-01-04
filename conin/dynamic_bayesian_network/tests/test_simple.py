import pytest
import pyomo.opt

from conin.util import try_import
from conin.dynamic_bayesian_network.inference import (
    inference_pyomo_map_query_DDBN,
)

from . import examples

with try_import() as pgmpy_available:
    import pgmpy
    from conin.common.pgmpy import convert_pgmpy_to_conin

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


#
# simple0
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple0_ALL_conin():
    """
    Z
    """
    G = examples.simple0_DDBN_conin()
    q = {("Z", 0): 1, ("Z", 1): 0}

    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple0_DDBN1_ALL_pgmpy():
    """
    Z
    """
    G = examples.simple0_DDBN1_pgmpy()
    q = {("Z", 0): 1, ("Z", 1): 0}

    G = convert_pgmpy_to_conin(G)
    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple0_DDBN2_ALL_pgmpy():
    """
    Z
    """
    G = examples.simple0_DDBN2_pgmpy()
    q = {("Z", 0): 1, ("Z", 1): 0}

    G = convert_pgmpy_to_conin(G)
    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


#
# simple1
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    G = convert_pgmpy_to_conin(G)
    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_simple1_B_conin():
#    """
#    A -> B
#
#    Evidence: A_0 = 1
#    """
#    G = examples.simple1_DDBN_conin()
#    q = {
#        ("A", 1): 0,
#        ("B", 0): 0,
#        ("B", 1): 1,
#    }
#
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(
#            pgm=G, evidence={("A", 0): 1}
#        )  # variables=None, evidence=None
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q == results.solution.variable_value


#@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_simple1_B_pgmpy():
#    """
#    A -> B
#
#    Evidence: A_0 = 1
#    """
#    G = examples.simple1_DDBN_pgmpy()
#    q = {
#        ("A", 1): 0,
#        ("B", 0): 0,
#        ("B", 1): 1,
#    }
#
#    G = convert_pgmpy_to_conin(G)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(
#            pgm=G, evidence={("A", 0): 1}
#        )  # variables=None, evidence=None
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q == results.solution.variable_value


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, solver=mip_solver)
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, solver=mip_solver)
    assert q == results.solution.variable_value


#
# simple2
#


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    G = convert_pgmpy_to_conin(G)
    results = inference_pyomo_map_query_DDBN(pgm=G, solver=mip_solver) # variables=None, evidence=None
    assert q == results.solution.variable_value


#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_simple2_B_conin():
#    """
#    A -> B
#
#    Evidence: A_0 = 1
#    """
#    G = examples.simple2_DDBN_conin()
#    q = {
#        ("A", 1): 0,
#        ("B", 0): 0,
#        ("B", 1): 1,
#    }
#
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(
#            pgm=G, evidence={("A", 0): 1}
#        )  # variables=None, evidence=None
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q == results.solution.variable_value


#@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_simple2_B_pgmpy():
#    """
#    A -> B
#
#    Evidence: A_0 = 1
#    """
#    G = examples.simple2_DDBN_pgmpy()
#    q = {
#        ("A", 1): 0,
#        ("B", 0): 0,
#        ("B", 1): 1,
#    }
#
#    G = convert_pgmpy_to_conin(G)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(
#            pgm=G, evidence={("A", 0): 1}
#        )  # variables=None, evidence=None
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q == results.solution.variable_value


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, solver=mip_solver)
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
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

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, solver=mip_solver)
    assert q == results.solution.variable_value
