"""Solver configuration.

``conin.config.default_mip_solver`` is the MIP solver that the
``"integer_program"`` method uses when no ``solver`` is given: the first
available one of gurobi, glpk, highs and appsi_highs, or ``None`` if there is
none.  It is looked up the first time it is used (not when conin is
imported), and then kept.
"""

import functools

import pyomo.environ as pyo
import pyomo.opt


@functools.cache
def _find_mip_solver():
    # check_available_solvers only detects legacy-style solvers; new-style
    # solvers (e.g. highs, appsi_highs) must be checked via SolverFactory
    # directly.
    solvers = pyomo.opt.check_available_solvers("gurobi", "glpk")
    if solvers:
        return solvers[0]
    for name in ("highs", "appsi_highs"):
        try:
            if pyo.SolverFactory(name).available():
                return name
        except Exception:
            pass
    return None


def mip_solver(solver=None):
    """Return ``solver``, or the default MIP solver if it is ``None``.

    Raises
    ------
    RuntimeError
        If ``solver`` is ``None`` and no MIP solver is available.
    """
    if solver is not None:
        return solver
    solver = _find_mip_solver()
    if solver is None:
        raise RuntimeError(
            "No MIP solver is available (looked for gurobi, glpk, highs and "
            "appsi_highs); install one or pass solver="
        )
    return solver


def __getattr__(name):
    if name == "default_mip_solver":
        return _find_mip_solver()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
