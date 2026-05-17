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

with try_import() as or_topas_available:
    import or_topas

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None
gurobi_available = len(pyomo.opt.check_available_solvers("gurobi")) > 0

import pytest

skipif_no_mip_solvers = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)
skipif_ortopas_not_available = pytest.mark.skipif(
    not or_topas_available, reason="or_topas not installed"
)
skipif_gurobi_not_available = pytest.mark.skipif(
    not gurobi_available, reason="gurobi not installed"
)
skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)


# ===============================================================================
#
# MIP tests
#
# ===============================================================================


@skipif_no_mip_solvers
def test_ABC_conin_mip():
    """
    Testing ABC using the conin repn with Pyomo MIP solvers
    """
    example = examples.ABC_conin()
    q = example.solutions[0].states

    results = inference_pyomo_map_query_MN(pgm=example.pgm, solver=mip_solver)
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_no_mip_solvers
def test_ABC_pgmpy_mip():
    """
    Testing ABC using the pgmpy repn with Pyomo MIP solvers
    """
    example = examples.ABC_pgmpy()
    q = example.solutions[0].states

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_pyomo_map_query_MN(pgm=pgm, solver=mip_solver)
    assert q == results.solution.states


@skipif_no_mip_solvers
def test2_ABC_conin_mip():
    """
    Testing ABC using the conin repn with Pyomo MIP solvers
    Specifying evidence.
    """
    example = examples.ABC_conin()
    evidence = {"B": 0}

    # Results without evidence values
    q = {"A": 2, "C": 1}
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, solver=mip_solver, evidence=evidence
    )
    assert q == results.solution.states

    # Results with evidence values
    q = {"A": 2, "B": 0, "C": 1}
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm,
        solver=mip_solver,
        evidence=evidence,
        solution_with_evidence=True,
    )
    assert q == results.solution.states


@skipif_no_mip_solvers
def test_ABC_constrained_pyomo_conin_mip():
    """
    Testing ABC using the conin repn with Pyomo MIP solvers
    Constrain the inference to ensure that all variables have different values.
    """
    example = examples.ABC_constrained_pyomo_conin()

    # without evidence
    q = example.solutions[0].states
    results = inference_pyomo_map_query_MN(pgm=example.pgm, solver=mip_solver)
    assert q == results.solution.states

    # with evidence
    q = {"A": 2, "C": 1}
    evidence = {"B": 0}
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, solver=mip_solver, evidence={"B": 0}
    )
    assert results.solution.states == {"A": 2, "C": 1}


# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================


@skipif_toulbar2_not_available
def test_ABC_conin_toulbar2():
    """
    Testing ABC using the conin repn with toulbar2 solver
    """
    example = examples.ABC_conin()
    results = inference_toulbar2_map_query_MN(pgm=example.pgm)
    assert results.solution.states == example.solutions[0].states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_ABC_pgmpy_toulbar2():
    """
    Testing ABC using the pgmpy repn with toulbar2 solver
    """
    example = examples.ABC_pgmpy()
    q = example.solutions[0].states

    pgm = convert_pgmpy_to_conin(example.pgm)
    results = inference_toulbar2_map_query_MN(pgm=pgm)
    assert q == results.solution.states


@skipif_toulbar2_not_available
def test2_ABC_conin_toulbar2():
    """
    Testing ABC using the conin repn with toulbar2 solver
    Specifying evidence.
    """
    example = examples.ABC_conin()
    q = example.solutions[0].states
    evidence = {"B": 0}

    # Results without evidence values
    q = {"A": 2, "C": 1}
    results = inference_toulbar2_map_query_MN(
        pgm=example.pgm,
        evidence=evidence,
    )
    assert q == results.solution.states

    # Results with evidence values
    q = {"A": 2, "B": 0, "C": 1}
    results = inference_toulbar2_map_query_MN(
        pgm=example.pgm,
        evidence=evidence,
        solution_with_evidence=True,
    )
    assert q == results.solution.states


@skipif_toulbar2_not_available
def test_ABC_constrained_toulbar2_conin():
    """
    Testing ABC using the conin repn with toulbar2 solver
    Constrain the inference to ensure that all variables have different values.
    """
    example = examples.ABC_constrained_toulbar2_conin()

    # without evidence
    q = example.solutions[0].states
    results = inference_toulbar2_map_query_MN(pgm=example.pgm)
    assert q == results.solution.states

    # with evidence
    q = {"A": 2, "C": 1}
    results = inference_toulbar2_map_query_MN(pgm=example.pgm, evidence={"B": 0})
    assert q == results.solution.states

    # with evidence and include evidence in output
    q = {"A": 2, "B": 0, "C": 1}
    results = inference_toulbar2_map_query_MN(
        pgm=example.pgm, evidence={"B": 0}, solution_with_evidence=True
    )
    assert q == results.solution.states


# ===============================================================================
#
# OR-Topas tests
#
# ===============================================================================


@skipif_no_mip_solvers
@skipif_ortopas_not_available
def test_ABC_conin_topas_balas_ask_1_solution():
    example = examples.ABC_conin()
    q = example.solutions[0].states

    solver_options = dict(
        num_solutions=1,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver=mip_solver,
        solver_options={},
        pool_manager=None,
        topas_method="balas",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 1
    assert q == results.solution.states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC_conin_topas_gurobi_ask_1_solution():
    example = examples.ABC_conin()
    q = example.solutions[0].states

    solver_options = dict(
        num_solutions=1,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver_options={},
        pool_manager=None,
        topas_method="gurobi_solution_pool",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 1
    assert q == results.solution.states


@skipif_no_mip_solvers
@skipif_ortopas_not_available
def test_ABC2_conin_topas_balas_ask_2_solution():
    example = examples.ABC2_conin()
    q = example.solutions[0].states

    solver_options = dict(
        num_solutions=2,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver=mip_solver,
        solver_options={},
        pool_manager=None,
        topas_method="balas",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 2
    assert q == results.solution.states
    assert results.solutions[0].states == example.solutions[0].states
    assert results.solutions[1].states == example.solutions[1].states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC2_conin_topas_gurobi_ask_2_solution():
    example = examples.ABC2_conin()
    solver_options = dict(
        num_solutions=2,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver_options={},
        pool_manager=None,
        topas_method="gurobi_solution_pool",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 2
    assert results.solution.states == example.solutions[0].states
    assert results.solutions[0].states == example.solutions[0].states
    assert results.solutions[1].states == example.solutions[1].states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC2_conin_topas_gurobi_ask_2_with_opt_gap_solution():
    example = examples.ABC2_conin()
    solver_options = dict(
        num_solutions=2,
        rel_opt_gap=None,
        abs_opt_gap=0,
        solver_options={},
        pool_manager=None,
        topas_method="gurobi_solution_pool",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 1
    assert results.solution.states == example.solutions[0].states
    assert results.solutions[0].states == example.solutions[0].states


@skipif_ortopas_not_available
@skipif_no_mip_solvers
def test_ABC_constrained_pyomo_conin_topas_balas_ask_1_solution():
    example = examples.ABC2_constrained_pyomo_conin()
    solver_options = dict(
        num_solutions=1,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver=mip_solver,
        solver_options={},
        pool_manager=None,
        topas_method="balas",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 1
    assert results.solution.states == example.solutions[0].states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC2_constrained_pyomo_conin_topas_balas_ask_2_solution():
    example = examples.ABC2_constrained_pyomo_conin()
    solver_options = dict(
        num_solutions=2,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver_options={},
        pool_manager=None,
        topas_method="balas",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 2
    assert results.solution.states == example.solutions[0].states
    assert results.solutions[0].states == example.solutions[0].states
    assert results.solutions[1].states == example.solutions[1].states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC2_constrained_pyomo_conin_topas_gurobi_ask_1_solution():
    example = examples.ABC2_constrained_pyomo_conin()
    solver_options = dict(
        num_solutions=1,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver_options={},
        pool_manager=None,
        topas_method="gurobi_solution_pool",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 1
    assert results.solution.states == example.solutions[0].states
    assert results.solutions[0].states == example.solutions[0].states


@skipif_gurobi_not_available
@skipif_ortopas_not_available
def test_ABC2_constrained_pyomo_conin_topas_gurobi_ask_2_solution():
    example = examples.ABC2_constrained_pyomo_conin()
    solver_options = dict(
        num_solutions=2,
        rel_opt_gap=None,
        abs_opt_gap=None,
        solver_options={},
        pool_manager=None,
        topas_method="gurobi_solution_pool",
    )
    results = inference_pyomo_map_query_MN(
        pgm=example.pgm, tee=False, solver="or_topas", solver_options=solver_options
    )
    assert len(results.solutions) == 2
    assert results.solution.states == example.solutions[0].states
    assert results.solutions[0].states == example.solutions[0].states
    assert results.solutions[1].states == example.solutions[1].states
