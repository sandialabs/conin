"""MAP queries on the weather DBN (unrolled over 0..4), with and without evidence.

The integer program, toulbar2 and variable elimination are exact, so they
must agree.
"""

import pytest
import pyomo.opt

from conin.util import try_import

from conin.examples import dynamic_bayesian_network as examples
from conin.algorithms.inference import map_query

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

needs_mip = pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
needs_toulbar2 = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)


def _methods(constraint_family):
    """(method, options, example factory) for the unconstrained or constrained weather DBN."""
    if constraint_family is None:
        factories = dict.fromkeys(
            ["integer_program", "toulbar2", "variable_elimination"],
            examples.weather_conin,
        )
    else:
        factories = {
            "integer_program": examples.weather_constrained_pyomo_conin,
            "toulbar2": examples.weather_constrained_toulbar2_conin,
            "variable_elimination": examples.weather_constrained_oracle_conin,
        }
    return [
        pytest.param(
            "integer_program",
            lambda: dict(solver=mip_solver),
            factories["integer_program"],
            marks=needs_mip,
            id="integer_program",
        ),
        pytest.param(
            "toulbar2",
            dict,
            factories["toulbar2"],
            marks=needs_toulbar2,
            id="toulbar2",
        ),
        pytest.param(
            "variable_elimination",
            dict,
            factories["variable_elimination"],
            id="variable_elimination",
        ),
    ]


# ===============================================================================
#
# No evidence: the documented solutions
#
# ===============================================================================


@pytest.mark.parametrize("method,options,factory", _methods(None))
def test_weather_A(method, options, factory):
    example = factory()
    results = map_query(example.pgm, method=method, stop=4, **options())
    assert results.solution.states == example.solutions[0].states


@pytest.mark.parametrize("method,options,factory", _methods("constrained"))
def test_weather_A_constrained(method, options, factory):
    """Exactly two rainy days."""
    example = factory()
    results = map_query(example.pgm, method=method, stop=4, **options())
    assert results.solution.states == example.solutions[0].states


# ===============================================================================
#
# With evidence on the observations O and H
#
# ===============================================================================

evidence_B = {
    ("O", 0): "Wet",
    ("O", 1): "Wet",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("H", 0): "Medium",
    ("H", 1): "Medium",
    ("H", 2): "Medium",
    ("H", 3): "Medium",
    ("H", 4): "Medium",
}

q_weather_B = {
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Mild",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Cloudy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}

q_weather_B_constrained = {
    ("T", 0): "Hot",
    ("T", 1): "Mild",
    ("T", 2): "Cold",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Rainy",
    ("W", 1): "Rainy",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


def _hidden(states):
    return {node: state for node, state in states.items() if node[0] in "TW"}


@pytest.mark.parametrize("method,options,factory", _methods(None))
def test_weather_B(method, options, factory):
    results = map_query(
        factory().pgm, method=method, stop=4, evidence=evidence_B, **options()
    )
    assert _hidden(results.solution.states) == q_weather_B


@pytest.mark.parametrize("method,options,factory", _methods("constrained"))
def test_weather_B_constrained(method, options, factory):
    results = map_query(
        factory().pgm, method=method, stop=4, evidence=evidence_B, **options()
    )
    assert _hidden(results.solution.states) == q_weather_B_constrained
