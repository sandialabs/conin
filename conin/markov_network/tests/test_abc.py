import pytest
import os

from math import log
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.markov_network import (
    create_MN_pyomo_map_query_model,
    solve_pyomo_map_query_model,
    create_MN_toulbar2_map_query_model,
    solve_toulbar2_map_query_model,
)
from conin.markov_network.factor_repn import State
from . import examples

with try_import() as pgmpy_available:
    import pgmpy
    from conin.common.pgmpy import convert_pgmpy_to_conin

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_pyomo_conin():
    pgm = examples.ABC_conin()
    model = create_MN_pyomo_map_query_model(pgm=pgm)
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_pyomo_pgmpy():
    pgm = examples.ABC_pgmpy()
    pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_pyomo_map_query_model(pgm=pgm)
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC3_pyomo_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_pyomo_map_query_model(pgm=pgm, variables=["A"])
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 2}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC4_pyomo_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_pyomo_map_query_model(pgm=pgm, variables=["B"])
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"B": 2}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC5_pyomo_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_pyomo_map_query_model(pgm=pgm, variables=["C"])
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def Xtest_ABC6_pyomo_conin():
    pgm = examples.ABC_conin()
    # pgm = convert_pgmpy_to_conin(pgm)
    model = create_MN_pyomo_map_query_model(pgm=pgm, variables=["C"], evidence={"B": 0})
    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"C": 1}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_constrained_pyomo():
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
    model = create_MN_pyomo_map_query_model(pgm=pgm)

    def diff_(M, s):
        s = State(s)
        return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1

    model.diff = pyo.Constraint([0, 1, 2], rule=diff_)

    results = solve_pyomo_map_query_model(model, solver=mip_solver)
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    # Setup constraints using the ConstrainedDiscreteMarkovNetwork class
    cpgm = examples.ABC_constrained_pyomo_conin()
    results = solve_pyomo_map_query_model(
        create_MN_pyomo_map_query_model(pgm=cpgm), solver=mip_solver
    )
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}


@pytest.mark.skipif(not pytoulbar2_available, reason="Toulbar2 not installed")
def test_ABC_toulbar2():
    pgm = examples.ABC_conin()
    model = create_MN_toulbar2_map_query_model(pgm=pgm)
    results = solve_toulbar2_map_query_model(model)
    assert results.solution.variable_value == {"A": 2, "B": 2, "C": 1}


@pytest.mark.skipif(not pytoulbar2_available, reason="Toulbar2 not installed")
def test_ABC_constrained_toulbar2():
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
    model = create_MN_toulbar2_map_query_model(pgm=pgm)

    for i in [0, 1, 2]:
        model.AddGeneralizedLinearConstraint(
            [(model.X["A"], i, 1), (model.X["B"], i, 1), (model.X["C"], i, 1)], "<=", 1
        )

    results = solve_toulbar2_map_query_model(model)
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    # Setup constraints using the ConstrainedDiscreteMarkovNetwork class
    cpgm = examples.ABC_constrained_toulbar2_conin()
    results = solve_toulbar2_map_query_model(
        create_MN_toulbar2_map_query_model(pgm=cpgm)
    )
    assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}
