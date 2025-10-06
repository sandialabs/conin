import pytest

from math import log
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.markov_network import (
    create_MN_map_query_pyomo_model,
    optimize_map_query_model,
)
from conin.markov_network.factor_repn import State
from . import examples

with try_import() as pgmpy_available:
    import pgmpy
    from conin.common.pgmpy import convert_pgmpy_to_conin

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_conin():
    pgm = examples.ABC_conin()
    model = create_MN_map_query_pyomo_model(pgm=pgm)
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_pgmpy():
    pgm = examples.ABC_pgmpy()
    pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_map_query_pyomo_model(pgm=pgm)
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC3_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_map_query_pyomo_model(pgm=pgm, variables=["A"])
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC4_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_map_query_pyomo_model(pgm=pgm, variables=["B"])
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"B": 2}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC5_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_map_query_pyomo_model(pgm=pgm, variables=["C"])
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC6_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_map_query_pyomo_model(pgm=pgm, variables=["C"], evidence={"B": 0})
    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_constrained():
    """
    Three variables with pair-wise interactions.

    The interactions have equal weights.  The unconstrained MAP solution is A:2, B:2, C:1.
    However, we include a constraint that excludes variable assignments to values that are equal.

    The constrained MAP solution is A:0, B:2, C:1.
    """
    # Constrain the inference to ensure that all variables have different
    # values

    # Explicit setup of constraints, which requires indexing using State() objects
    pgm = examples.ABC_conin()
    model = create_MN_map_query_pyomo_model(pgm=pgm)

    def diff_(M, s):
        s = State(s)
        return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1

    model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

    results = optimize_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    # Setup constraints using the ConstrainedDiscreteMarkovNetwork class, which does not require State() functions
    cpgm = examples.ABC_constrained_conin()
    results = optimize_map_query_model(
        create_MN_map_query_pyomo_model(pgm=cpgm), solver=mip_solver
    )
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}
