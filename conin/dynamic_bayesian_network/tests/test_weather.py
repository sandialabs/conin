import pytest

from conin.util import try_import
from conin.dynamic_bayesian_network import (
    create_DDBN_map_query_model,
    optimize_map_query_model,
)

from . import examples

with try_import() as pgmpy_available:
    import pgmpy


def test_weather_A_conin():
    """
    Test with conin representation
    """
    pgm = examples.weather_conin()
    q = {
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

    model = create_DDBN_map_query_model(
        pgm=pgm, stop=4
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather1_A_pgmpy():
    """
    Test with pgmpy TabularCPD representation
    """
    pgm = examples.weather1_pgmpy()
    q = {
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

    model = create_DDBN_map_query_model(
        pgm=pgm, stop=4
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather2_A_pgmpy():
    """
    Test with pgmpy MapCPD representation
    """
    pgm = examples.weather2_pgmpy()
    q = {
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

    model = create_DDBN_map_query_model(
        pgm=pgm, stop=4
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_weather_B_conin():
    """
    Test with conin representation

    Using evidence.
    """
    pgm = examples.weather_conin()
    evidence = {
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
    # TODO - confirm this answer makes sense
    q = {
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

    model = create_DDBN_map_query_model(pgm=pgm, stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather1_B_pgmpy():
    """
    Test with pgmpy TabularCPD representation

    Using evidence.
    """
    pgm = examples.weather1_pgmpy()
    evidence = {
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
    # TODO - confirm this answer makes sense
    q = {
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

    model = create_DDBN_map_query_model(pgm=pgm, stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather2_B_pgmpy():
    """
    Test with pgmpy MapCPD representation

    Using evidence.
    """
    pgm = examples.weather2_pgmpy()
    evidence = {
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
    # TODO - confirm this answer makes sense
    q = {
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

    model = create_DDBN_map_query_model(pgm=pgm, stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_weather_A_constrained_conin():
    """
    Test with conin representation
    """
    pgm = examples.weather_constrained_conin()
    q = {
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

    model = create_DDBN_map_query_model(
        pgm=pgm, stop=4
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather_A_constrained_pgmpy():
    """
    Test with pgmpy representation
    """
    pgm = examples.weather_constrained_pgmpy()
    q = {
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

    model = create_DDBN_map_query_model(
        pgm=pgm, stop=4
    )  # variables=None, evidence=None
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


def test_weather_B_constrained_conin():
    """
    Test with pgmpy representation

    Using evidence.
    """
    cpgm = examples.pgmpy_weather_constrained_conin()
    evidence = {
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
    q = {
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

    model = cpgm.create_map_query_model(stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value


@pytest.mark.skipif(not pgmpy_available, reason="pgmpy not installed")
def test_weather_B_constrained_pgmpy():
    """
    Test with pgmpy representation

    Using evidence.
    """
    cpgm = examples.pgmpy_weather_constrained_pgmpy()
    evidence = {
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
    q = {
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

    model = cpgm.create_map_query_model(stop=4, evidence=evidence)
    results = optimize_map_query_model(model, solver="glpk")
    assert q == results.solution.variable_value
