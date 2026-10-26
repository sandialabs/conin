"""Solve the HMM network-flow integer program (``ip_formulation="network_flow"``).

The model is built by the ``"pyomo_network_flow"`` representation
(:mod:`conin.representations.pyomo.hmm_network_flow`).
"""

import logging
import math

from conin.config import default_mip_solver
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.exceptions import InsufficientSolutionsError, InvalidInputError
from conin.graphical_models.hmm_observations import as_sequence
from conin.inference.hmm_results import key_solutions_by_time
from conin.inference.results import map_results, map_solution, score_hmm_solutions
from conin.util import try_import

import pyomo.environ as pyo
from pyomo.common.timing import tic, toc, TicTocTimer
from pyomo.contrib.alternative_solutions.aos_utils import (
    get_model_variables,
)

with try_import() as or_topas_available:
    import or_topas.aos as aos

logger = logging.getLogger("conin")


def _solve_network_flow(repn, q):
    """Solve a MAP query on the ``"pyomo_network_flow"`` representation of an HMM.

    ``q`` is the query of :func:`~conin.inference.map_query`.  Without
    constraints the model is solved with :func:`lp_inference`, with
    constraints with :func:`ip_inference`; ``q.options`` are their options
    (``use_cut_generation`` is ignored with a warning).
    """
    options = dict(q.options)
    if options.pop("use_cut_generation", None):
        logger.warning(
            "Cut generation is not supported for the HMM 'network_flow' integer programming formulation"
        )
    observed = as_sequence(
        q.evidence,
        "evidence (the observations) of the 'network_flow' integer programming formulation",
    )
    infer = _ip_inference if repn.constraints else _lp_inference
    results = infer(repn, observed=observed, **options)
    if isinstance(q.evidence, dict):
        key_solutions_by_time(results)
    return results


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
        ``solution``, ``solutions``, ``termination_condition`` and
        ``solvetime``; the solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``) and
        ``solver_objective`` (the LP objective).  See
        :mod:`conin.inference.results`.
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
    if num_solutions != 1:
        raise InvalidInputError(
            "lp_inference finds one solution; num_solutions must be 1, got "
            f"{num_solutions}"
        )

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

    objective = pyo.value(M.hmm.o)
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in data.hmm.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = hmm.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    ans = map_results(
        [map_solution(hidden, solver_objective=objective)], solvetime=solvetime
    )
    score_hmm_solutions(ans, hmm, observed)
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
        ``solution``, ``solutions``, ``termination_condition`` and
        ``solvetime``; each solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``),
        ``solver_objective`` (the IP objective) and ``variables`` (the
        nonzero model variables).  See :mod:`conin.inference.results`.
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

        if not aos_pm.solutions:
            raise InsufficientSolutionsError(
                "No solutions found by the 'or_topas' solver"
            )
        solutions = [
            parse_aos_solution_pyomo_ip_inference(
                aos_solution=aos_solution, M=M, hmm=hmm, T=T, data=data
            )
            for aos_solution in aos_pm.solutions
        ]
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

        # We are maximizing, so the lower bound is the best incumbent found
        # by the solver.
        objective = res["Problem"][0]["Lower bound"]

        solutions = [
            parse_model_solution_pyomo_ip_inference(
                M=M, hmm=hmm, T=T, objective=objective, data=data
            )
        ]

    if debug:
        print(f"E: {len(data.hmm.E)}")
        print(f"F: {len(data.hmm.F)}")
        print(f"Gt: {len(data.hmm.Gt)}")
        print(f"Ge: {len(data.hmm.Ge)}")
        print(f"GG: {len(data.hmm.GG)}")
        print(f"FF: {len(data.hmm.FF)}")
        print(f"T: {len(data.hmm.T)}")
        print(f"A: {len(data.hmm.A)}")

    results = map_results(solutions, solvetime=solvetime)
    return score_hmm_solutions(results, hmm, observed)


def parse_model_solution_pyomo_ip_inference(M, hmm, T, objective, *, data):
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

    return map_solution(hidden, solver_objective=objective, variables=variables)


def parse_aos_solution_pyomo_ip_inference(aos_solution, M, hmm, T, *, data):
    obj_list = aos_solution._objectives
    assert len(obj_list) > 0, "Solution in parse_aos_solution has empty objective list"
    # this implicitly forces an assumption that there is only 1 objective
    objective = obj_list[0].value
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

    return map_solution(hidden, solver_objective=objective, variables=variables)
