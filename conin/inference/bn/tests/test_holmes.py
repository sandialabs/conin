import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.inference.bn import (
    inference_pyomo_map_query_BN,
    inference_toulbar2_map_query_BN,
)

from conin.bayesian_network import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


# ===============================================================================
#
# conin tests for holmes example
#
# ===============================================================================

# TODO: Add tests with evidence.


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes0_pyomo_conin():
    example = examples.holmes_conin()
    variables = None
    evidence = None

    results = inference_pyomo_map_query_BN(
        pgm=example.pgm,
        variables=variables,
        evidence=evidence,
        solver=mip_solver,
    )
    assert results.solution.variable_value == example.solution


# ===============================================================================
#
# toulbar2 tests for holmes example
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_holmes0_toulbar2_conin():
    example = examples.holmes_conin()
    variables = None
    evidence = None

    results = inference_toulbar2_map_query_BN(
        pgm=example.pgm,
        variables=variables,
        evidence=evidence,
    )
    assert results.solution.variable_value == example.solution


# ===============================================================================
#
# pgmpy tests for holmes example
#
# ===============================================================================


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes0_pgmpy():
    example = examples.holmes_pgmpy()
    variables = None
    evidence = None

    infer = VariableElimination(example.pgm)
    assert infer.map_query(variables=variables, evidence=evidence) == example.solution

    from conin.common.pgmpy import convert_pgmpy_to_conin

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        solver=mip_solver,
    )
    assert results.solution.variable_value == example.solution


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes1():
    example = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"W": "w", "G": "g"}

    infer = VariableElimination(example.pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    #    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes2():
    example = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"A": "-a"}

    infer = VariableElimination(example.pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    #    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes3():
    example = examples.holmes_pgmpy()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"A": "a"}

    infer = VariableElimination(example.pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    #    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes4():
    example = examples.holmes_pgmpy()
    q = {"W": "w", "G": "-g"}
    variables = ["W", "G"]
    evidence = {"B": "b"}

    infer = VariableElimination(example.pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    #    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes5():
    example = examples.holmes_pgmpy()
    q = {"B": "-b"}
    variables = ["B"]
    evidence = {"R": "r", "W": "-w"}

    infer = VariableElimination(example.pgm)
    assert q == infer.map_query(variables=variables, evidence=evidence)

    # pgm = convert_pgmpy_to_conin(pgm)
    # with pytest.raises(RuntimeError):
    #    model = create_pyomo_map_query_model_BN(
    #        pgm=pgm, variables=variables, evidence=evidence
    #    )
    #    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    #    assert q == results.solution.variable_value
