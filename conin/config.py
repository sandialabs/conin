import os
import pyomo.opt

conin_home = os.environ.get("conin_HOME", None)

_solvers = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
default_mip_solver = _solvers[0] if _solvers else None
