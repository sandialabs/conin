import pytest
import pyomo.opt

from conin.util import try_import

from conin.examples import dynamic_bayesian_network as examples
from conin.algorithms.inference import map_query

with try_import() as pytoulbar2_available:
    import pytoulbar2

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None


# ===============================================================================
#
# pyomo tests
#
# ===============================================================================


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather_pyomo_A_conin():
    """
    Test with conin representation
    """
    example = examples.weather_conin()
    results = map_query(
        example.pgm,
        method="integer_program",
        stop=4,
        solver=mip_solver,
    )  # variables=None, evidence=None
    assert results.solution.states == example.solutions[0].states


# TODO - confirm this answer makes sense
# q_weather_B = {
#    ("T", 0): "Hot",
#    ("T", 1): "Hot",
#    ("T", 2): "Hot",
#    ("T", 3): "Hot",
#    ("T", 4): "Hot",
#    ("W", 0): "Cloudy",
#    ("W", 1): "Cloudy",
#    ("W", 2): "Cloudy",
#    ("W", 3): "Cloudy",
#    ("W", 4): "Cloudy",
# }


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_weather_B_conin():
#    """
#    Test with conin representation
#
#    Using evidence.
#    """
#    pgm = examples.weather_conin()
#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(pgm=pgm, stop=4, evidence=evidence)
#        results = map_query(model, method="integer_program", solver=mip_solver)
#        assert q_weather_B == results.solution.states


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather_pyomo_A_constrained_conin():
    """
    Test with conin representation
    """
    example = examples.weather_constrained_pyomo_conin()
    results = map_query(
        example.pgm, method="integer_program", stop=4, solver=mip_solver
    )  # variables=None, evidence=None
    assert results.solution.states == example.solutions[0].states


# q_weather_B_constrained = {
#    ("T", 0): "Hot",
#    ("T", 1): "Mild",
#    ("T", 2): "Cold",
#    ("T", 3): "Hot",
#    ("T", 4): "Hot",
#    ("W", 0): "Rainy",
#    ("W", 1): "Rainy",
#    ("W", 2): "Sunny",
#    ("W", 3): "Sunny",
#    ("W", 4): "Sunny",
# }


# ===============================================================================
#
# toulbar2 tests
#
# ===============================================================================


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def test_weather_toulbar2_A_conin():
    """
    Test with conin representation
    """
    example = examples.weather_conin()
    results = map_query(
        example.pgm,
        method="toulbar2",
        stop=4,
    )  # variables=None, evidence=None
    assert results.solution.states == example.solutions[0].states


# TODO - confirm this answer makes sense
# q_weather_B = {
#    ("T", 0): "Hot",
#    ("T", 1): "Hot",
#    ("T", 2): "Hot",
#    ("T", 3): "Hot",
#    ("T", 4): "Hot",
#    ("W", 0): "Cloudy",
#    ("W", 1): "Cloudy",
#    ("W", 2): "Cloudy",
#    ("W", 3): "Cloudy",
#    ("W", 4): "Cloudy",
# }


# @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
# def test_weather_B_conin():
#    """
#    Test with conin representation
#
#    Using evidence.
#    """
#    pgm = examples.weather_conin()
#    evidence = {
#        ("O", 0): "Wet",
#        ("O", 1): "Wet",
#        ("O", 2): "Dry",
#        ("O", 3): "Dry",
#        ("O", 4): "Dry",
#        ("H", 0): "Medium",
#        ("H", 1): "Medium",
#        ("H", 2): "Medium",
#        ("H", 3): "Medium",
#        ("H", 4): "Medium",
#    }
#
#    with pytest.raises(RuntimeError):
#        model = create_toulbar2_map_query_model_DDBN(pgm=pgm, stop=4, evidence=evidence)
#        results = map_query(model, method="toulbar2")
#        assert q_weather_B == results.solution.states


@pytest.mark.skipif(not pytoulbar2_available, reason="pytoulbar2 not installed")
def Xtest_weather_toulbar2_A_constrained_conin():
    """
    Test with conin representation
    """
    example = examples.weather_constrained_toulbar2_conin()
    results = map_query(
        example.pgm,
        method="toulbar2",
        stop=4,
    )  # variables=None, evidence=None
    assert results.solution.states == example.solutions[0].states


# q_weather_B_constrained = {
#    ("T", 0): "Hot",
#    ("T", 1): "Mild",
#    ("T", 2): "Cold",
#    ("T", 3): "Hot",
#    ("T", 4): "Hot",
#    ("W", 0): "Rainy",
#    ("W", 1): "Rainy",
#    ("W", 2): "Sunny",
#    ("W", 3): "Sunny",
#    ("W", 4): "Sunny",
# }
