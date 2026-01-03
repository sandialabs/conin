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
# pyomo tests for cancer - All variables
#
# ===============================================================================


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer1_conin_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    example = examples.cancer1_BN_conin()
    results = inference_pyomo_map_query_BN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer1_pgmpy_ALL():
    """
    Cancer model from pgmpy examples (using TabularCPD)

    No evidence
    """
    example = examples.cancer1_BN_pgmpy()

    infer = VariableElimination(example.pgm)
    assert (
        infer.map_query(variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"])
        == example.solution
    )

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_BN(pgm=pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer2_pgmpy_ALL():
    """
    Cancer model from pgmpy examples (using MapCPD)

    No evidence
    """
    example = examples.cancer2_BN_pgmpy()

    infer = VariableElimination(example.pgm)
    assert (
        infer.map_query(variables=["Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"])
        == example.solution
    )

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_BN(pgm=pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# ===============================================================================
#
# toulbar2 tests for cancer - All variables
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_cancer1_toulbar2_ALL():
    """
    Cancer model from pgmpy examples

    No evidence
    """
    example = examples.cancer1_BN_conin()
    results = inference_toulbar2_map_query_BN(pgm=example.pgm)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_cancer1_conin_Cancer():
#    """
#    Cancer model from pgmpy examples
#
#    Evidence for Cancer
#    """
#    pgm = examples.cancer1_BN_conin()
#    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}
#
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"Cancer": 0}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value


# @pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_cancer1_pgmpy_Cancer():
#    """
#    Cancer model from pgmpy examples
#
#    Evidence for Cancer
#    """
#    pgm = examples.cancer1_BN_pgmpy()
#    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 0, "Pollution": 0}
#
#    infer = VariableElimination(pgm)
#    assert q == infer.map_query(
#        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#        evidence={"Cancer": 0},
#    )
#
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"Cancer": 0}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_cancer1_conin_ALL_constrained1():
#    """
#    Cancer model from pgmpy examples
#
#    No evidence
#    Constrained inference of Xray and Dyspnoea so they are different
#    """
#    pgm = examples.cancer1_BN_conin()
#
#    model = create_pyomo_map_query_model_BN(pgm=pgm)  # variables=None, evidence=None
#    model.c = pyo.ConstraintList()
#    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
#    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
#
#    results = solve_pyomo_map_query_model(model, solver=mip_solver)  # num=1
#    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
#    assert q == results.solution.variable_value


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer1_conin_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    example = examples.cancer1_BN_constrained_conin()
    results = inference_pyomo_map_query_BN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_cancer2_pgmpy_Cancer():
#    """
#    Cancer model from pgmpy examples
#
#    Evidence for Cancer
#    """
#    pgm = examples.cancer2_BN_pgmpy()
#    q = {"Xray": 0, "Dyspnoea": 0, "Smoker": 1, "Pollution": 0}
#
#    infer = VariableElimination(pgm)
#    assert q == infer.map_query(
#        variables=["Dyspnoea", "Pollution", "Smoker", "Xray"],
#        evidence={"Cancer": 0},
#    )
#
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_BN(
#            pgm=pgm, evidence={"Cancer": 0}
#        )  # variables=None, evidence=None
#        results = solve_pyomo_map_query_model(model, solver=mip_solver)
#        assert q == results.solution.variable_value


# @pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_cancer2_pgmpy_ALL_constrained1():
#    """
#    Cancer model from pgmpy examples
#
#    No evidence
#    Constrained inference of Xray and Dyspnoea so they are different
#    """
#    pgm = examples.cancer2_BN_pgmpy()
#    q = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
#
#    pgm = convert_pgmpy_to_conin(pgm)
#    model = create_pyomo_map_query_model_BN(pgm=pgm)  # variables=None, evidence=None
#    model.c = pyo.ConstraintList()
#    model.c.add(model.X["Dyspnoea", 1] + model.X["Xray", 1] <= 1)
#    model.c.add(model.X["Dyspnoea", 0] + model.X["Xray", 0] <= 1)
#
#    results = solve_pyomo_map_query_model(model, solver=mip_solver)  # num=1
#    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer2_pgmpy_ALL_constrained2():
    """
    Cancer model from pgmpy examples

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    example = examples.cancer2_BN_constrained_pgmpy()
    results = inference_pyomo_map_query_BN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution
