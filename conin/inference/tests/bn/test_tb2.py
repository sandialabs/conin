import pyomo.opt

from conin.util import try_import
from conin.examples import bayesian_network as examples
from conin.inference import map_query

with try_import() as pgmpy_available:
    from pgmpy.inference import VariableElimination

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

import pytest

skipif_no_mip_solvers = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
)
skipif_pgmpy_not_available = pytest.mark.skipif(
    not pgmpy_available, reason="pgmpy not installed"
)
skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)


# ===============================================================================
#
# Pyomo tests
#
# ===============================================================================


@skipif_no_mip_solvers
def test_tb2_BN_conin_mip():
    example = examples.tb2_BN_conin()
    q = example.solutions[0].states
    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_no_mip_solvers
def test_tb2_BN_pgmpy_mip():
    from conin.pgmpy import convert_pgmpy_to_conin

    example = examples.tb2_BN_pgmpy()
    q = example.solutions[0].states
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = map_query(pgm, method="integer_program", solver=mip_solver)
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_no_mip_solvers
def test_tb2_BN_pgmpy_mapcpd_mip():
    from conin.pgmpy import convert_pgmpy_to_conin

    example = examples.tb2_BN_pgmpy_mapcpd()
    q = example.solutions[0].states
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = map_query(pgm, method="integer_program", solver=mip_solver)
    assert q == results.solution.states


# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================


@skipif_toulbar2_not_available
def test_tb2_BN_conin_toulbar2():
    example = examples.tb2_BN_conin()
    q = example.solutions[0].states
    results = map_query(example.pgm, method="toulbar2")
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_tb2_BN_pgmpy_toulbar2():
    from conin.pgmpy import convert_pgmpy_to_conin

    example = examples.tb2_BN_pgmpy()
    q = example.solutions[0].states
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = map_query(pgm, method="toulbar2")
    assert q == results.solution.states


@skipif_pgmpy_not_available
@skipif_toulbar2_not_available
def test_tb2_BN_pgmpy_mapcpd_toulbar2():
    from conin.pgmpy import convert_pgmpy_to_conin

    example = examples.tb2_BN_pgmpy_mapcpd()
    q = example.solutions[0].states
    pgm = convert_pgmpy_to_conin(example.pgm)
    results = map_query(pgm, method="toulbar2")
    assert q == results.solution.states
