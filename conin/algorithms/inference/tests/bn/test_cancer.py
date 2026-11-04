import pytest
import pyomo.opt

from conin.util import try_import
from conin.examples import bayesian_network as examples
from conin.algorithms.inference import map_query

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None


# ===============================================================================
#
# pyomo tests for cancer - All variables
#
# ===============================================================================


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer1_conin_ALL():
    """
    The cancer model (Korb and Nicholson)

    No evidence
    """
    example = examples.cancer1_BN_conin()
    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# toulbar2 tests for cancer - All variables
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_cancer1_toulbar2_ALL():
    """
    The cancer model (Korb and Nicholson)

    No evidence
    """
    example = examples.cancer1_BN_conin()
    results = map_query(example.pgm, method="toulbar2")
    assert results.solution.states == example.solutions[0].states


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_cancer1_pyomo_conin_ALL_constrained2():
    """
    The cancer model (Korb and Nicholson)

    No evidence
    Constrained inference of Xray and Dyspnoea so they are different
    """
    example = examples.cancer1_BN_constrained_pyomo_conin()
    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states
