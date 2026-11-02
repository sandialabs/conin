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
# conin tests for holmes example
#
# ===============================================================================

# TODO: Add tests with evidence.


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_holmes0_pyomo_conin():
    example = examples.holmes_conin()
    variables = None
    evidence = None
    q = example.solutions[0].states

    results = map_query(
        example.pgm,
        method="integer_program",
        variables=variables,
        evidence=evidence,
        solver=mip_solver,
    )
    assert q == results.solution.states


# ===============================================================================
#
# toulbar2 tests for holmes example
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_holmes0_toulbar2_conin():
    example = examples.holmes_conin()
    variables = None
    evidence = None

    results = map_query(
        example.pgm,
        method="toulbar2",
        variables=variables,
        evidence=evidence,
    )
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# Partial MAP queries with variable elimination
#
# ===============================================================================


@pytest.mark.parametrize(
    "variables,evidence,expected",
    [
        (["B"], {"W": "w", "G": "g"}, {"B": "-b"}),
        (["B"], {"A": "-a"}, {"B": "-b"}),
        (["W", "G"], {"A": "a"}, {"W": "w", "G": "-g"}),
        (["W", "G"], {"B": "b"}, {"W": "w", "G": "-g"}),
        (["B"], {"R": "r", "W": "-w"}, {"B": "-b"}),
    ],
)
def test_holmes_variable_elimination(variables, evidence, expected):
    results = map_query(
        examples.holmes_conin().pgm,
        method="variable_elimination",
        variables=variables,
        evidence=evidence,
    )
    assert results.solution.states == expected
