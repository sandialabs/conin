import math
import munch

# from conin.hmm import hmm_application
from conin.hmm import ConstrainedHiddenMarkovModel

import pyomo.environ as pyo
from pyomo.common.timing import tic, toc
from pyomo.contrib.alternative_solutions.aos_utils import (
    get_model_variables,
    get_active_objective,
)


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
    num_solutions=1,
    observed,
    solver="gurobi",
    solver_options=None,
    debug=False,
    quiet=True,
):
    assert (
        num_solutions == 1
    ), "ERROR: Support for inferring multiple solutions with an IP model has not been setup"

    if debug:
        tic("Generating Model - START")
    M = hmm.chmm.generate_model(observed=observed)
    data = hmm.chmm.data
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

    #
    # We are maximizing, so the lower bound is the best incumbent found by the solver.
    # We assume that the active objective is a log-likelihood score.
    #
    log_likelihood = res["Problem"][0]["Lower bound"]

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

    soln = munch.Munch(
        variable_value=hidden,
        hidden=hidden,
        log_likelihood=log_likelihood,
        variables=variables,
    )
    ans = munch.Munch(
        observed=observed,
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
    )
    if debug:
        ans.hmm = hmm.chmm.hmm
        ans.M = M
        print(f"E: {len(data.hmm.E)}")
        print(f"F: {len(data.hmm.F)}")
        print(f"Gt: {len(data.hmm.Gt)}")
        print(f"Ge: {len(data.hmm.Ge)}")
        print(f"GG: {len(data.hmm.GG)}")
        print(f"FF: {len(data.hmm.FF)}")
        print(f"T: {len(data.hmm.T)}")
        print(f"A: {len(data.hmm.A)}")
    return ans
