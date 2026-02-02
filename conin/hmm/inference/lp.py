import math
import munch

# from conin.hmm import hmm_application
from conin.hmm import ConstrainedHiddenMarkovModel
from conin.util import try_import

import pyomo.environ as pyo
from pyomo.common.timing import tic, toc
from pyomo.contrib.alternative_solutions.aos_utils import (
    get_model_variables,
    get_active_objective,
)

with try_import() as or_topas_available:
    import or_topas.aos as aos


def lp_inference(
    *,
    hmm,
    observed,
    num_solutions=1,
    solver="gurobi",
    solver_options=None,
    debug=False,
    quiet=True,
):
    assert (
        num_solutions == 1
    ), "ERROR: Support for inferring multiple solutions with an LP model has not been setup"

    if debug:
        tic("Generating Model - START")
    chmm = ConstrainedHiddenMarkovModel(hmm=hmm)
    chmm.initialize_chmm("pyomo")
    M = chmm.chmm.generate_unconstrained_model(observed=observed)
    data = chmm.chmm.data
    if debug:
        toc("Generating Model - STOP")
    if debug:
        tic("Optimizing Model - START")
    opt = pyo.SolverFactory(solver)
    res = opt.solve(M, tee=not quiet)
    pyo.assert_optimal_termination(res)
    if debug:
        toc("Optimizing Model - STOP")

    T = len(observed)

    log_likelihood = pyo.value(M.hmm.o)
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = hmm.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    soln = munch.Munch(
        variable_value=hidden, hidden=hidden, log_likelihood=log_likelihood
    )
    ans = munch.Munch(
        observed=observed,
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
    )
    if debug:
        ans.hmm = hmm
        ans.M = M
        print(f"E: {len(data.hmm.E)}")
        print(f"F: {len(data.hmm.F)}")
        print(f"G: {len(data.hmm.G)}")
        print(f"GG: {len(data.hmm.GG)}")
        print(f"FF: {len(data.hmm.FF)}")
        print(f"T: {len(data.hmm.T)}")
        print(f"A: {len(data.hmm.A)}")
    return ans


def ip_inference(
    *,
    hmm,
    observed,
    solver="gurobi",
    solver_options=None,
    debug=False,
    quiet=True,
):
    if debug:
        tic("Generating Model - START")
    M = hmm.chmm.generate_model(observed=observed)
    data = hmm.chmm.data
    T = len(observed)
    if debug:
        toc("Generating Model - STOP")
    if solver == "or_topas":
        if not or_topas_available:
            raise RuntimeError("or_topas Solver Unavailable")
        if solver_options == None:
            solver_options = dict()
        topas_method = solver_options.pop("topas_method", "balas")
        if debug:
            tic("Optimizing Model with OR_TOPAS - START")
        if topas_method == "balas":
            aos_pm = aos.enumerate_binary_solutions(M, **solver_options)
        elif topas_method == "gurobi_solution_pool":
            aos_pm = aos.gurobi_generate_solutions(M, **solver_options)
        else:
            raise RuntimeError(f"Asked for {topas_method=}, which is not supported")
        if debug:
            toc("Optimizing Model with OR_TOPAS - STOP")
        assert len(aos_pm.solutions) > 0, f"No solutions found for OR_TOPAS Solver use"
        solutions = []
        for index, aos_solution in enumerate(aos_pm.solutions):
            aos_sol_munch = parse_aos_solution_pyomo_ip_inference(
                aos_solution=aos_solution, M=M, hmm=hmm, T=T
            )
            if index == 0:
                first_sol = aos_sol_munch
            solutions.append(aos_sol_munch)
        termination_condition = "ok"
        soln = first_sol
    else:
        # this is solver != "or_topas"
        if debug:
            tic("Optimizing Model - START")
        opt = pyo.SolverFactory(solver)
        res = opt.solve(M, tee=not quiet)
        pyo.assert_optimal_termination(res)
        if debug:
            toc("Optimizing Model - STOP")

        # We are maximizing, so the lower bound is the best incumbent found by the solver.
        # We assume that the active objective is a log-likelihood score.
        # so this is just the objective
        log_likelihood = res["Problem"][0]["Lower bound"]

        soln = parse_model_solution_pyomo_ip_inference(
            M=M, hmm=hmm, T=T, log_likelihood=log_likelihood
        )
        solutions = [soln]
    if debug:
        print(f"E: {len(data.hmm.E)}")
        print(f"F: {len(data.hmm.F)}")
        print(f"Gt: {len(data.hmm.Gt)}")
        print(f"Ge: {len(data.hmm.Ge)}")
        print(f"GG: {len(data.hmm.GG)}")
        print(f"FF: {len(data.hmm.FF)}")
        print(f"T: {len(data.hmm.T)}")
        print(f"A: {len(data.hmm.A)}")
    return munch.Munch(
        observed=observed,
        solution=soln,
        solutions=solutions,
        termination_condition="ok",
    )


def parse_model_solution_pyomo_ip_inference(M, hmm, T, log_likelihood):
    data = hmm.chmm.data
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = hmm.hidden_markov_model.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    model_variables = get_model_variables(M, include_fixed=True)
    variables = {
        str(v): pyo.value(v) for v in model_variables if math.fabs(pyo.value(v)) > 1e-3
    }

    return munch.Munch(
        variable_value=hidden,
        hidden=hidden,
        log_likelihood=log_likelihood,
        variables=variables,
    )


def parse_aos_solution_pyomo_ip_inference(aos_solution, M, hmm, T):
    obj_list = aos_solution._objectives
    assert len(obj_list) > 0, "Solution in parse_aos_solution has empty objective list"
    # this implicitly forces an assumption that there is only 1 objective
    log_likelihood = obj_list[0].value
    data = hmm.chmm.data
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            aos_var_hmm_x_t_a = aos_solution.variable(M.hmm.x[t, a].name)
            if aos_var_hmm_x_t_a.value > 0.5:
                hidden[t] = hmm.hidden_markov_model.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    model_variables = aos_solution._variables
    variables = {
        aos_var.name: aos_var.value
        for aos_var in model_variables
        if math.fabs(aos_var.value) > 1e-3
    }

    return munch.Munch(
        variable_value=hidden,
        hidden=hidden,
        log_likelihood=log_likelihood,
        variables=variables,
    )
