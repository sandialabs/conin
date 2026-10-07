import os
import pyomo.opt
import pyomo.environ as _pyo

conin_home = os.environ.get("conin_HOME", None)

# check_available_solvers only detects legacy-style solvers; new-style solvers
# (e.g. highs, appsi_highs) must be checked via SolverFactory directly.
_solvers = pyomo.opt.check_available_solvers("gurobi", "glpk")
if not _solvers:
    for _name in ("highs", "appsi_highs"):
        try:
            _s = _pyo.SolverFactory(_name)
            if _s.available():
                _solvers = [_name]
                break
        except Exception:
            pass
default_mip_solver = _solvers[0] if _solvers else None
