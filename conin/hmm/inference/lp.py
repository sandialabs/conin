import math
import munch

from conin.hmm import hmm_application

import pyomo.environ as pyo
from pyomo.common.timing import tic, toc
from pyomo.contrib.alternative_solutions.aos_utils import get_model_variables


def lp_inference(
    *,
    statistical_model,
    num_solutions=1,
    observed,
    solver="gurobi",
    solver_options=None,
    debug=False,
    quiet=True,
):
    assert (
        num_solutions == 1
    ), "ERROR: Support for inferring multiple solutions with an LP model has not been setup"

    assert isinstance(
        statistical_model, hmm_application.HMMApplication
    ), "ERROR: LP inference is only supported with a HMMApplication instance"
    algebraic_hmm = statistical_model.algebraic

    if debug:
        tic("Generating Model - START")
    M = algebraic_hmm.generate_unconstrained_model(observations=observed)
    if debug:
        toc("Generating Model - STOP")
    if debug:
        tic("Optimizing Model - START")
    opt = pyo.SolverFactory(solver)
    res = opt.solve(M, tee=not quiet)
    # TODO - Check termination condition here
    if debug:
        toc("Optimizing Model - STOP")

    T = len(observed)

    log_likelihood = pyo.value(M.hmm.o)
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in algebraic_hmm.data.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = algebraic_hmm.hmm.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    ans = munch.Munch(
        observations=observed,
        solutions=[munch.Munch(hidden=hidden, log_likelihood=log_likelihood)],
        termination_condition="ok",
    )
    if debug:
        ans.hmm = algebraic_hmm.hmm
        ans.M = M
        print(f"E: {len(algebraic_hmm.data.E)}")
        print(f"F: {len(algebraic_hmm.data.F)}")
        print(f"G: {len(algebraic_hmm.data.G)}")
        print(f"GG: {len(algebraic_hmm.data.GG)}")
        print(f"FF: {len(algebraic_hmm.data.FF)}")
        print(f"T: {len(algebraic_hmm.data.T)}")
        print(f"A: {len(algebraic_hmm.data.A)}")
    return ans


def ip_inference(
    *,
    statistical_model,
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

    assert isinstance(
        statistical_model, hmm_application.HMMApplication
    ), "ERROR: IP inference is only supported with a HMMApplication instance"
    algebraic_hmm = statistical_model.algebraic

    if debug:
        tic("Generating Model - START")
    M = algebraic_hmm.generate_model(observations=observed)
    if debug:
        toc("Generating Model - STOP")
    if debug:
        tic("Optimizing Model - START")
    opt = pyo.SolverFactory(solver)
    res = opt.solve(M, tee=not quiet)
    # TODO - check terminaton_condition here
    if debug:
        toc("Optimizing Model - STOP")

    T = len(observed)

    log_likelihood = pyo.value(M.hmm.o)
    hidden = ["__UNKNOWN__"] * T
    for t in range(T):
        for a in algebraic_hmm.data.A:
            if pyo.value(M.hmm.x[t, a]) > 0.5:
                hidden[t] = algebraic_hmm.hmm.hidden_to_external[a]
        assert (
            hidden[t] != "__UNKNOWN__"
        ), f"ERROR: Unexpected missing hidden state at time step {t}"

    model_variables = get_model_variables(M)
    variables = {
        str(v): pyo.value(v) for v in model_variables if math.fabs(pyo.value(v)) > 1e-3
    }

    ans = munch.Munch(
        observations=observed,
        solutions=[
            munch.Munch(
                hidden=hidden, log_likelihood=log_likelihood, variables=variables
            )
        ],
        termination_condition="ok",
    )
    if debug:
        ans.hmm = algebraic_hmm.hmm
        ans.M = M
        print(f"E: {len(algebraic_hmm.data.E)}")
        print(f"F: {len(algebraic_hmm.data.F)}")
        print(f"G: {len(algebraic_hmm.data.G)}")
        print(f"GG: {len(algebraic_hmm.data.GG)}")
        print(f"FF: {len(algebraic_hmm.data.FF)}")
        print(f"T: {len(algebraic_hmm.data.T)}")
        print(f"A: {len(algebraic_hmm.data.A)}")
    return ans
