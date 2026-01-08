import pytest
import pyomo.environ as pyo
import pyomo.opt

from conin.util import try_import
from conin.inference.mn import (
    inference_pyomo_map_query_MN,
    inference_toulbar2_map_query_MN,
)
from conin.markov_network import examples

with try_import() as pgmpy_available:
    import pgmpy
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
def test_ABC_pyomo_conin():
    example = examples.ABC_conin()
    results = inference_pyomo_map_query_MN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_ABC_pyomo_pgmpy():
    example = examples.ABC_pgmpy()
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_MN(pgm=pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def Xtest_ABC3_pyomo_conin():
#    pgm = examples.ABC_conin()
#    results = inference_pyomo_map_query_model_MN(pgm=pgm, variables=["A"], solver=mip_solver)
#    assert results.solution.variable_value == {"A": 2}


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def Xtest_ABC4_pyomo_conin():
#    pgm = examples.ABC_conin()
#    results = inference_pyomo_map_query_model_MN(pgm=pgm, variables=["B"], solver=mip_solver)
#    assert results.solution.variable_value == {"B": 2}


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def Xtest_ABC5_pyomo_conin():
#    pgm = examples.ABC_conin()
#    results = inference_pyomo_map_query_model_MN(pgm=pgm, variables=["C"], solver=mip_solver)
#    assert results.solution.variable_value == {"C": 1}


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def Xtest_ABC6_pyomo_conin():
#    pgm = examples.ABC_conin()
#    # pgm = convert_pgmpy_to_conin(pgm)
#    model = create_pyomo_map_query_model_MN(pgm=pgm, variables=["C"], evidence={"B": 0})
#    results = solve_pyomo_map_query_model(model, solver=mip_solver)
#    assert results.solution.variable_value == {"C": 1}


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
    # pgm = examples.ABC_conin()
    # model = create_pyomo_map_query_model_MN(pgm=pgm)
    #
    # def diff_(M, s):
    #    s = State(s)
    #    return M.x["A", s] + M.x["B", s] + M.x["C", s] <= 1
    #
    # model.diff = pyo.Constraint([0, 1, 2], rule=diff_)
    #
    # results = solve_pyomo_map_query_model(model, solver=mip_solver)
    # assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    # Setup constraints using the ConstrainedDiscreteMarkovNetwork class
    example = examples.ABC_constrained_pyomo_conin()
    results = inference_pyomo_map_query_MN(pgm=example.pgm, solver=mip_solver)
    assert results.solution.variable_value == example.solution


# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="Toulbar2 not installed")
def test_ABC_toulbar2():
    example = examples.ABC_conin()
    results = inference_toulbar2_map_query_MN(pgm=example.pgm)
    assert results.solution.variable_value == example.solution


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
    # pgm = examples.ABC_conin()
    # model = create_toulbar2_map_query_model_MN(pgm=pgm)
    #
    # for i in [0, 1, 2]:
    #    model.AddGeneralizedLinearConstraint(
    #        [(model.X["A"], i, 1), (model.X["B"], i, 1), (model.X["C"], i, 1)], "<=", 1
    #    )
    #
    # results = solve_toulbar2_map_query_model(model)
    # assert results.solution.variable_value == {"A": 0, "B": 2, "C": 1}

    # Setup constraints using the ConstrainedDiscreteMarkovNetwork class
    example = examples.ABC_constrained_toulbar2_conin()
    results = inference_toulbar2_map_query_MN(pgm=example.pgm)
    assert results.solution.variable_value == example.solution
