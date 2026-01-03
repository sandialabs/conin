import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.bayesian_network.inference import (
    inference_pyomo_map_query_BN,
    inference_toulbar2_map_query_BN,
)
from . import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination
    from conin.common.pgmpy import convert_pgmpy_to_conin

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None

# ===============================================================================
#
# Pyomo tests
#
# ===============================================================================


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_pyomo_ALL_conin():
    """
    A -> B
    """
    example = examples.simple1_BN_conin()

    results = inference_pyomo_map_query_BN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_simple1_B_conin():
#    """
#    A -> B, with evidence for A
#    """
#    example = examples.simple1_BN_conin()
#    q = {"B": 0}
#
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"A": 1}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_pyomo_ALL_pgmpy():
    """
    A -> B
    """
    example = examples.simple1_BN_pgmpy()

    infer = VariableElimination(example.pgm)
    assert infer.map_query(variables=["A", "B"]) == example.solution

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_BN(pgm=pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_simple1_B_pgmpy():
#    """
#    A -> B, with evidence for A
#    """
#    pgm = examples.simple1_BN_pgmpy()
#    q = {"B": 0}
#
#    infer = VariableElimination(pgm)
#    assert q == infer.map_query(variables=["B"], evidence={"A": 1})
#
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"A": 1}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple2_pyomo_ALL_pgmpy():
    """
    A -> B
    """
    example = examples.simple2_BN_pgmpy()

    infer = VariableElimination(example.pgm)
    assert infer.map_query(variables=["A", "B"]) == example.solution

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_BN(pgm=pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_simple2_B_pgmpy():
#    """
#    A -> B, with evidence for A
#    """
#    example = examples.simple2_BN_pgmpy()
#    q = {"B": 0}
#
#    infer = VariableElimination(pgm)
#    assert q == infer.map_query(variables=["B"], evidence={"A": 1})
#
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"A": 1}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value

# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_simple1_toulbar2_ALL():
    """
    A -> B
    """
    example = examples.simple1_BN_conin()

    results = inference_toulbar2_map_query_BN(pgm=example.pgm)
    assert results.solution.variable_value == example.solution
