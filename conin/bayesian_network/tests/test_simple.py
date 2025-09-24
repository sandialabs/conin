import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.bayesian_network import (
    create_BN_map_query_model,
    optimize_map_query_model,
)

from . import examples

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination
    from conin.common.pgmpy import convert_pgmpy_to_conin

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_ALL_conin():
    """
    A -> B
    """
    pgm = examples.simple1_BN_conin()
    q = {"A": 0, "B": 1}

    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver=mip_solver)
    assert q == results.solution.variable_value


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_B_conin():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple1_BN_conin()
    q = {"B": 0}

    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"A": 1}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver=mip_solver)
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_ALL_pgmpy():
    """
    A -> B
    """
    pgm = examples.simple1_BN_pgmpy()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    pgm = convert_pgmpy_to_conin(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver=mip_solver)
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple1_B_pgmpy():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple1_BN_pgmpy()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    pgm = convert_pgmpy_to_conin(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"A": 1}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver=mip_solver)
        assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple2_ALL_pgmpy():
    """
    A -> B
    """
    pgm = examples.simple2_BN_pgmpy()
    q = {"A": 0, "B": 1}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["A", "B"])

    pgm = convert_pgmpy_to_conin(pgm)
    model = create_BN_map_query_model(pgm=pgm)  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver=mip_solver)
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_simple2_B_pgmpy():
    """
    A -> B, with evidence for A
    """
    pgm = examples.simple2_BN_pgmpy()
    q = {"B": 0}

    infer = VariableElimination(pgm)
    assert q == infer.map_query(variables=["B"], evidence={"A": 1})

    pgm = convert_pgmpy_to_conin(pgm)
    with pytest.raises(RuntimeError):
        model = create_BN_map_query_model(
            pgm=pgm, evidence={"A": 1}
        )  # variables=None, evidence=None
        results = optimize_map_query_model(model, solver=mip_solver)
        assert q == results.solution.variable_value
