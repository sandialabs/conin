import pytest
import pyomo.opt

from conin.util import try_import

from conin.examples import bayesian_network as examples
from conin.algorithms.inference import map_query

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None


# ===============================================================================
#
# pyomo DBDA_51 tests
#
# ===============================================================================


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_DBDA_51_pyomo():
    example = examples.DBDA_5_1_conin()
    results = map_query(example.pgm, method="integer_program", solver=mip_solver)
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# toulbar2 DBDA_51 tests
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_DBDA_51_toulbar2():
    example = examples.DBDA_5_1_conin()
    results = map_query(example.pgm, method="toulbar2")
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# Partial MAP queries with variable elimination
#
# ===============================================================================


@pytest.mark.parametrize(
    "evidence",
    [
        {"test-result1": 0},
        {"test-result1": 0, "test-result2": 1},
        {"test-result1": 0, "test-result2": 0},
    ],
    ids=["one_positive", "one_positive_and_one_negative", "two_positive"],
)
def test_DBDA_51_variable_elimination(evidence):
    results = map_query(
        examples.DBDA_5_1_conin().pgm,
        method="variable_elimination",
        variables=["disease-state"],
        evidence=evidence,
    )
    assert results.solution.states == {"disease-state": 1}
