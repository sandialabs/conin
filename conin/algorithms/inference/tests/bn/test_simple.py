import pyomo.opt

from conin.util import try_import
from conin.examples import bayesian_network as examples


from conin.algorithms.inference import map_query

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

import pytest

skipif_no_mip_solvers = pytest.mark.skipif(
    not mip_solver, reason="No mip solver installed"
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
def test1_simple1_BN_conin_mip():
    """
    A -> B

    Solve with mip
    """
    example = examples.simple1_BN_conin()
    q = example.solutions[0].states

    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert q == results.solution.states


@skipif_no_mip_solvers
def test2_simple1_BN_conin_mip():
    """
    A -> B, with evidence for A

    Solve with mip
    """
    example = examples.simple1_BN_conin()
    q = {"B": 0}

    results = map_query(
        example.pgm, method="integer_program", solver=mip_solver, evidence={"A": 1}
    )
    assert q == results.solution.states


# ===============================================================================
#
# Toulbar2 tests
#
# ===============================================================================


@skipif_toulbar2_not_available
def test1_simple1_BN_conin_toulbar2():
    """
    A -> B

    Solve with toulbar2
    """
    example = examples.simple1_BN_conin()
    q = example.solutions[0].states

    results = map_query(example.pgm, method="toulbar2")
    assert q == results.solution.states


@skipif_toulbar2_not_available
def test2_simple1_BN_conin_toulbar2():
    """
    A -> B, with evidence for A

    Solve with toulbar2
    """
    example = examples.simple1_BN_conin()
    q = {"B": 0}

    results = map_query(example.pgm, method="toulbar2", evidence={"A": 1})
    assert q == results.solution.states
