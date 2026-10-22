"""Solve the HMM network-flow integer program (``ip_formulation="network_flow"``).

The model is built by the ``"pyomo_network_flow"`` representation
(:mod:`conin.representations.pyomo.hmm_network_flow`).
"""

import math
import munch

from conin.config import default_mip_solver
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.util import try_import

import pyomo.environ as pyo
from pyomo.common.timing import tic, toc, TicTocTimer
from pyomo.contrib.alternative_solutions.aos_utils import (
    get_model_variables,
)

with try_import() as or_topas_available:
    import or_topas.aos as aos


def lp_inference(
    *,
    hmm,
    observed,
    num_solutions=1,
    solver=default_mip_solver,
    solver_options=None,
    debug=False,
    quiet=True,
):
    """Solve the network-flow model of an unconstrained HMM.

    Parameters
    ----------
    hmm : HiddenMarkovModel
    observed : list
        Observed labels.
    num_solutions : int, optional
        Must be ``1``.
    solver, solver_options, debug, quiet
        The Pyomo solver name, (unused) options, debug output and solver
        output.

    Returns
    -------
    munch.Munch
        ``observed``, ``solution``, ``solutions`` (each with ``states``,
        ``hidden`` and ``log_likelihood``), ``termination_condition`` and
        ``solvetime``.
    """
    repn = ConstrainedHiddenMarkovModel(hmm).representation("pyomo_network_flow")
    return _lp_inference(
        repn,
        observed=observed,
        num_solutions=num_solutions,
        solver=solver,
        solver_options=solver_options,
        debug=debug,
        quiet=quiet,
    )


def _lp_inference(
    repn,
    *,
    observed,
    num_solutions=1,
    solver=default_mip_solver,
    solver_options=None,
    debug=False,
    quiet=True,
):
    """:func:`lp_inference` on a ``"pyomo_network_flow"`` representation."""
    assert (
        num_solutions == 1
    ), "ERROR: Support for inferring multiple solutions with an LP model has not been setup"

    hmm = repn.hidden_markov_model
    if debug:
        tic("Generating Model - START")
    M, data = repn.create_model(observed=observed)
    if debug:
        toc("Generating Model - STOP")
    if debug:
        tic("Optimizing Model - START")
    opt = pyo.SolverFactory(solver)

    timer = TicTocTimer()
    timer.tic(None)
    res = opt.solve(M, tee=not quiet)
    solvetime = timer.toc(None)

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

    soln = munch.Munch(states=hidden, hidden=hidden, log_likelihood=log_likelihood)
    ans = munch.Munch(
        observed=observed,
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
        solvetime=solvetime,
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
    solver=default_mip_solver,
    solver_options=None,
    debug=False,
    quiet=True,
):
    """Solve the network-flow model of a constrained HMM.

    Parameters
    ----------
    hmm : ConstrainedHiddenMarkovModel
        Its constraints must be of family ``"pyomo"`` or ``"algebraic"`` (or
        none).
    observed : list
        Observed labels.
    solver, solver_options, debug, quiet
        The Pyomo solver name (``"or_topas"`` enumerates several solutions),
        its options, debug output and solver output.

    Returns
    -------
    munch.Munch
        ``observed``, ``solution``, ``solutions`` (each with ``states``,
        ``hidden``, ``log_likelihood`` and ``variables``),
        ``termination_condition`` and ``solvetime``.
    """
    return _ip_inference(
        hmm.representation("pyomo_network_flow"),
        observed=observed,
        solver=solver,
        solver_options=solver_options,
        debug=debug,
        quiet=quiet,
    )


def _ip_inference(
    repn,
    *,
    observed,
    solver=default_mip_solver,
    solver_options=None,
    debug=False,
    quiet=True,
):
    """:func:`ip_inference` on a ``"pyomo_network_flow"`` representation."""
    hmm = repn.hidden_markov_model
    if debug:
        tic("Generating Model - START")
    M, data = repn.create_model(observed=observed)
    T = len(observed)
    if debug:
        toc("Generating Model - STOP")
    if solver == "or_topas":
        if not or_topas_available:
            raise RuntimeError("or_topas Solver Unavailable")
        # Copy, so that popping topas_method leaves the caller's dict unchanged.
        solver_options = dict(solver_options or {})
        topas_method = solver_options.pop("topas_method", "balas")

        if debug:
            tic("Optimizing Model with OR_TOPAS - START")

        timer = TicTocTimer()
        timer.tic(None)
        if topas_method == "balas":
            aos_pm = aos.enumerate_binary_solutions(M, **solver_options)
        elif topas_method == "gurobi_solution_pool":
            aos_pm = aos.gurobi_generate_solutions(M, **solver_options)
        else:
            raise RuntimeError(f"Asked for {topas_method=}, which is not supported")
        solvetime = timer.toc(None)

        if debug:
            toc("Optimizing Model with OR_TOPAS - STOP")

        assert len(aos_pm.solutions) > 0, "No solutions found for OR_TOPAS Solver use"
        solutions = []
        for index, aos_solution in enumerate(aos_pm.solutions):
            aos_sol_munch = parse_aos_solution_pyomo_ip_inference(
                aos_solution=aos_solution, M=M, hmm=hmm, T=T, data=data
            )
            if index == 0:
                first_sol = aos_sol_munch
            solutions.append(aos_sol_munch)
        soln = first_sol
    else:
        # this is solver != "or_topas"
        if debug:
            tic("Optimizing Model - START")
        opt = pyo.SolverFactory(solver)

        timer = TicTocTimer()
        timer.tic(None)
        res = opt.solve(M, tee=not quiet)
        solvetime = timer.toc(None)

        pyo.assert_optimal_termination(res)
        if debug:
            toc("Optimizing Model - STOP")

        # We are maximizing, so the lower bound is the best incumbent found by the solver.
        # We assume that the active objective is a log-likelihood score.
        # so this is just the objective
        log_likelihood = res["Problem"][0]["Lower bound"]

        soln = parse_model_solution_pyomo_ip_inference(
            M=M, hmm=hmm, T=T, log_likelihood=log_likelihood, data=data
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
        solvetime=solvetime,
    )


def parse_model_solution_pyomo_ip_inference(M, hmm, T, log_likelihood, *, data):
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = hmm.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    model_variables = get_model_variables(M, include_fixed=True)
    variables = {
        str(v): pyo.value(v) for v in model_variables if math.fabs(pyo.value(v)) > 1e-3
    }

    return munch.Munch(
        states=hidden,
        hidden=hidden,
        log_likelihood=log_likelihood,
        variables=variables,
    )


def parse_aos_solution_pyomo_ip_inference(aos_solution, M, hmm, T, *, data):
    obj_list = aos_solution._objectives
    assert len(obj_list) > 0, "Solution in parse_aos_solution has empty objective list"
    # this implicitly forces an assumption that there is only 1 objective
    log_likelihood = obj_list[0].value
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            aos_var_hmm_x_t_a = aos_solution.variable(M.hmm.x[t, a].name)
            if aos_var_hmm_x_t_a.value > 0.5:
                hidden[t] = hmm.hidden_to_external[a]
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
