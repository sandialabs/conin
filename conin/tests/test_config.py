"""conin.config: the default MIP solver."""

import pytest

import conin.config as config


@pytest.fixture
def no_cache():
    config._find_mip_solver.cache_clear()
    yield
    config._find_mip_solver.cache_clear()


def test_default_mip_solver_is_looked_up_once(monkeypatch, no_cache):
    calls = []

    def check_available_solvers(*names):
        calls.append(names)
        return ["glpk"]

    monkeypatch.setattr(
        config.pyomo.opt, "check_available_solvers", check_available_solvers
    )
    assert config.default_mip_solver == "glpk"
    assert config.mip_solver() == "glpk"
    from conin.config import default_mip_solver

    assert default_mip_solver == "glpk"
    assert calls == [("gurobi", "glpk")]


def test_falls_back_to_highs(monkeypatch, no_cache):
    class Highs:
        def available(self):
            return True

    monkeypatch.setattr(config.pyomo.opt, "check_available_solvers", lambda *n: [])
    monkeypatch.setattr(config.pyo, "SolverFactory", lambda name: Highs())
    assert config.default_mip_solver == "highs"


def test_no_solver(monkeypatch, no_cache):
    class Unavailable:
        def available(self):
            return False

    monkeypatch.setattr(config.pyomo.opt, "check_available_solvers", lambda *n: [])
    monkeypatch.setattr(config.pyo, "SolverFactory", lambda name: Unavailable())
    assert config.default_mip_solver is None
    assert config.mip_solver("cbc") == "cbc"
    with pytest.raises(RuntimeError, match="No MIP solver is available"):
        config.mip_solver()


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'conin_home'"):
        config.conin_home
