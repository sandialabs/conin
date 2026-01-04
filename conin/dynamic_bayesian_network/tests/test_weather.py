import pytest
import pyomo.opt

from conin.util import try_import
from conin.dynamic_bayesian_network.inference import (
    inference_pyomo_map_query_DDBN
)

from . import examples

with try_import() as pgmpy_available:
    import pgmpy
    from conin.common.pgmpy import convert_pgmpy_to_conin

mip_solver = pyomo.opt.check_available_solvers("glpk", "gurobi")
mip_solver = mip_solver[0] if mip_solver else None


q_weather_A = {
    ("H", 0): "Low",
    ("H", 1): "Low",
    ("H", 2): "Low",
    ("H", 3): "Low",
    ("H", 4): "Low",
    ("O", 0): "Dry",
    ("O", 1): "Dry",
    ("O", 2): "Dry",
    ("O", 3): "Dry",
    ("O", 4): "Dry",
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Hot",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Sunny",
    ("W", 1): "Sunny",
    ("W", 2): "Sunny",
    ("W", 3): "Sunny",
    ("W", 4): "Sunny",
}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather_A_conin():
    """
    Test with conin representation
    """
    pgm = examples.weather_conin()

    results = inference_pyomo_map_query_DDBN(
        pgm=pgm, stop=4, solver=mip_solver,
    )  # variables=None, evidence=None
    assert q_weather_A == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather1_A_pgmpy():
    """
    Test with pgmpy TabularCPD representation
    """
    pgm = examples.weather1_pgmpy()

    pgm = convert_pgmpy_to_conin(pgm)
    results = inference_pyomo_map_query_DDBN(pgm=pgm, stop=4, solver=mip_solver) # variables=None, evidence=None
    assert q_weather_A == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather2_A_pgmpy():
    """
    Test with pgmpy MapCPD representation
    """
    pgm = examples.weather2_pgmpy()

    pgm = convert_pgmpy_to_conin(pgm)
    results = inference_pyomo_map_query_DDBN(pgm=pgm, stop=4, solver=mip_solver) # variables=None, evidence=None
    assert q_weather_A == results.solution.variable_value


# TODO - confirm this answer makes sense
q_weather_B = {
    ("T", 0): "Hot",
    ("T", 1): "Hot",
    ("T", 2): "Hot",
    ("T", 3): "Hot",
    ("T", 4): "Hot",
    ("W", 0): "Cloudy",
    ("W", 1): "Cloudy",
    ("W", 2): "Cloudy",
    ("W", 3): "Cloudy",
    ("W", 4): "Cloudy",
}


#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_weather_B_conin():
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
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q_weather_B == results.solution.variable_value


#@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_weather1_B_pgmpy():
#    """
#    Test with pgmpy TabularCPD representation
#
#    Using evidence.
#    """
#    pgm = examples.weather1_pgmpy()
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
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(pgm=pgm, stop=4, evidence=evidence)
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q_weather_B == results.solution.variable_value


#@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_weather2_B_pgmpy():
#    """
#    Test with pgmpy MapCPD representation
#
#    Using evidence.
#    """
#    pgm = examples.weather2_pgmpy()
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
#    pgm = convert_pgmpy_to_conin(pgm)
#    with pytest.raises(RuntimeError):
#        model = create_pyomo_map_query_model_DDBN(pgm=pgm, stop=4, evidence=evidence)
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q_weather_B == results.solution.variable_value


q_weather_A_constrained = {
    ("W", 0): "Sunny",
    ("T", 0): "Hot",
    ("O", 0): "Dry",
    ("H", 0): "Low",
    ("W", 1): "Sunny",
    ("T", 1): "Hot",
    ("O", 1): "Dry",
    ("H", 1): "Low",
    ("W", 2): "Sunny",
    ("T", 2): "Hot",
    ("O", 2): "Dry",
    ("H", 2): "Low",
    ("W", 3): "Rainy",
    ("T", 3): "Hot",
    ("O", 3): "Wet",
    ("H", 3): "High",
    ("W", 4): "Rainy",
    ("T", 4): "Mild",
    ("O", 4): "Wet",
    ("H", 4): "High",
}


@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather_A_constrained_conin():
    """
    Test with conin representation
    """
    cpgm = examples.weather_constrained_conin()

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, stop=4, solver=mip_solver) # variables=None, evidence=None
    assert q_weather_A_constrained == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
def test_weather_A_constrained_pgmpy():
    """
    Test with pgmpy representation
    """
    cpgm = examples.weather_constrained_pgmpy()

    results = inference_pyomo_map_query_DDBN(pgm=cpgm, stop=4, solver=mip_solver) # variables=None, evidence=None
    assert q_weather_A_constrained == results.solution.variable_value


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


#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_weather_B_constrained_conin():
#    """
#    Test with pgmpy representation
#
#    Using evidence.
#    """
#    cpgm = examples.weather_constrained_conin()
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
#        model = create_pyomo_map_query_model_DDBN(pgm=cpgm, stop=4, evidence=evidence)
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q_weather_B_constrained == results.solution.variable_value


#@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
#@pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
#def test_weather_B_constrained_pgmpy():
#    """
#    Test with pgmpy representation
#
#    Using evidence.
#    """
#    cpgm = examples.weather_constrained_pgmpy()
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
#        model = create_pyomo_map_query_model_DDBN(pgm=cpgm, stop=4, evidence=evidence)
#        results = inference_pyomo_map_query_DDBN(model, solver=mip_solver)
#        assert q_weather_B_constrained == results.solution.variable_value

