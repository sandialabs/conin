"""Generic solve and solution parsing for the Pyomo MAP query models.

The models are built by the ``"pyomo"`` representations
(:mod:`conin.representations.pyomo`); this module only solves them.
"""

import munch
import pyomo.environ as pe
from pyomo.common.timing import TicTocTimer

from conin.config import default_mip_solver
from conin.representations.pyomo.common import C5LazyCutGenerator, has_omitted_c5
from conin.util import try_import

with try_import() as or_topas_available:
    import or_topas.aos as aos


def solve_pyomo_map_query_model(
    model,
    *,
    solver=default_mip_solver,
    tee=False,
    solution_with_fixed=False,
    solution_with_evidence=False,
    timing=False,
    solver_options=None,
    evidence=None,
    write_lp_file=None,
    c5_user_cuts=False,
    c5_cut_tol=1e-6,
    c5_max_cuts_per_node=100,
):
    """Solve a MAP query model.

    If the model was created with ``use_cut_generation=True`` and ``solver="gurobi"``,
    then the c5 constraints are generated as lazy constraints in a Gurobi
    callback (and optionally as user cuts if ``c5_user_cuts=True``).  For
    other solvers, the c5 constraints are added to the model before solving.
    """
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("optimize_map_query_model - START")

    cut_stats = None
    if has_omitted_c5(model) and solver != "gurobi":
        raise RuntimeError(
            f"The c5 constraints were omitted, but cut generation is not supported for solver={solver}."
        )

    if solver == "or_topas":
        #
        # AOS solver in or_topas
        #
        if not or_topas_available:  # pragma:nocover
            raise RuntimeError("or_topas not installed")

        if solver_options is None:
            solver_options = dict()
        topas_method = solver_options.pop("topas_method", "balas")

        if timing:  # pragma:nocover
            timer.toc("Initialize solver")

        solver_timer = TicTocTimer()
        solver_timer.tic(None)
        if topas_method == "balas":
            aos_pm = aos.enumerate_binary_solutions(model, **solver_options)
        elif topas_method == "gurobi_solution_pool":
            aos_pm = aos.gurobi_generate_solutions(model, tee=tee, **solver_options)
        else:  # pragma:nocover
            raise RuntimeError(f"Asked for {topas_method=}, which is not supported")
        solvetime = solver_timer.toc(None)

        if timing:  # pragma:nocover
            timer.toc("Completed optimization")

        assert len(aos_pm.solutions) > 0, "No solutions found by 'or_topas' solver"
        solutions = [
            parse_aos_solution_pyomo_map_query(model, aos_solution, solution_with_fixed)
            for aos_solution in aos_pm.solutions
        ]
        soln = solutions[0]
        termination_condition = "ok"

    elif solver == "gurobi":
        #
        # Gurobi ILP solver
        #
        if write_lp_file:
            model.write(write_lp_file)

        opt = pe.SolverFactory("gurobi_persistent")
        opt.set_instance(model)
        if solver_options:
            for key, val in solver_options.items():
                opt.set_gurobi_param(key, val)

        cut_generator = None
        if has_omitted_c5(model):
            cut_generator = C5LazyCutGenerator(
                model,
                user_cuts=c5_user_cuts,
                tol=c5_cut_tol,
                max_cuts_per_node=c5_max_cuts_per_node,
            )
            opt.set_gurobi_param("LazyConstraints", 1)
            if c5_user_cuts:
                # Allow cuts on the original model to be translated into the
                # presolved model
                opt.set_gurobi_param("PreCrush", 1)
            opt.set_callback(cut_generator)

        if timing:  # pragma:nocover
            timer.toc("Initialize solver")

        solver_timer = TicTocTimer()
        solver_timer.tic(None)
        res = opt.solve(tee=tee, save_results=False)
        solvetime = solver_timer.toc(None)

        if timing:  # pragma:nocover
            timer.toc("Completed optimization")

        pe.assert_optimal_termination(res)
        soln = parse_model_solution_pyomo_map_query(model, solution_with_fixed)
        solutions = [soln]
        termination_condition = "ok"
        if cut_generator is not None:
            cut_stats = munch.Munch(
                num_lazy_cuts=cut_generator.num_lazy_cuts,
                num_user_cuts=cut_generator.num_user_cuts,
            )

    else:
        #
        # Other ILP solver
        #
        if write_lp_file:
            model.write(write_lp_file)

        opt = pe.SolverFactory(solver)
        if solver_options:
            opt.options = solver_options

        if timing:  # pragma:nocover
            timer.toc("Initialize solver")

        solver_timer = TicTocTimer()
        solver_timer.tic(None)
        res = opt.solve(model, tee=tee)
        solvetime = solver_timer.toc(None)

        if timing:  # pragma:nocover
            timer.toc("Completed optimization")

        pe.assert_optimal_termination(res)
        soln = parse_model_solution_pyomo_map_query(model, solution_with_fixed)
        solutions = [soln]
        termination_condition = "ok"

    #
    # Post-optimization processing
    #
    if evidence and solution_with_evidence and not solution_with_fixed:
        for soln in solutions:
            for k, v in evidence.items():
                soln.states[k] = v

    if timing:  # pragma:nocover
        timer.toc("optimize_map_query_model - STOP")

    results = munch.Munch(
        solution=soln,
        solutions=solutions,
        termination_condition=termination_condition,
        solvetime=solvetime,
    )
    if cut_stats is not None:
        results.cut_stats = cut_stats
        if timing:  # pragma:nocover
            timer.toc(f"cut statistics: {cut_stats}")

    return results


def parse_model_solution_pyomo_map_query(model, solution_with_fixed):
    var = {}
    variables = set()
    fixed_variables = set()
    for r, s in model.V:
        variables.add(r)
        if model.V(r, s).is_fixed():
            fixed_variables.add(r)
            if solution_with_fixed and pe.value(model.V(r, s)) > 0.5:
                var[r] = s.value
        elif pe.value(model.V(r, s)) > 0.5:
            var[r] = s.value
    assert variables == set(var.keys()).union(
        fixed_variables
    ), "Some variables do not have values."

    soln = munch.Munch(states=var, log_factor_sum=pe.value(model.o))
    return soln


def parse_aos_solution_pyomo_map_query(model, aos_solution, solution_with_fixed):
    var = {}
    variables = set()
    fixed_variables = set()
    for r, s in model.V:
        variables.add(r)
        aos_var_X_r_s = aos_solution.variable(model.V(r, s).name)
        if aos_var_X_r_s.fixed:
            fixed_variables.add(r)
            if solution_with_fixed and aos_var_X_r_s.value > 0.5:
                # N.B. this works the same between parse_aos and parse_model
                #     because r and s are parameters not variables
                var[r] = s.value
        elif aos_var_X_r_s.value > 0.5:
            # N.B. this works the same between parse_aos and parse_model
            #     because r and s are parameters not variables
            var[r] = s.value
    assert variables == set(var.keys()).union(
        fixed_variables
    ), "Some variables do not have values."

    obj_list = aos_solution._objectives
    assert len(obj_list) > 0, "Solution in parse_aos_solution has empty objective list"
    soln = munch.Munch(states=var, log_factor_sum=obj_list[0].value)
    return soln
