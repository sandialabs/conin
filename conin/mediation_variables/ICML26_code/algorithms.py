import time
from typing import Tuple, List

import numpy as np
import torch

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    RangeSet,
    Var,
    maximize,
    value,
    SolverFactory,
)
from pyomo.opt import SolverStatus, TerminationCondition


def hmm_constrained_inference(
    observations: torch.Tensor,
    transition: torch.Tensor,
    emission: torch.Tensor,
    initial: torch.Tensor,
    N: int,
    solver_name: str = "gurobi",
    time_limit: float | None = None,
    mip_gap: float | None = None,
    tee: bool = False,
) -> Tuple[List, float, bool]:
    """
    Exact constrained HMM MAP inference via a dense flow-based MILP in Pyomo.

    Constraint:
        The hidden-state subsequence [0, 1, ..., N-1] must occur at least once
        somewhere in the state sequence.

    Formulation:
        - z[t,k] for all t,k
        - y[t,i,j] for all t,i,j
        - a[s] for all candidate start positions s

    Inputs are treated as probabilities/scores. For HMM MAP inference, this
    function optimizes the log joint score.

    Args:
        observations: torch.Tensor of shape (T,), integer observation symbols
        transition: torch.Tensor of shape (K, K)
        emission: torch.Tensor of shape (K, M)
        initial: torch.Tensor of shape (K,)
        N: required subsequence length
        solver_name: usually "gurobi"
        time_limit: optional time limit in seconds
        mip_gap: optional relative MIP gap
        tee: whether to stream solver output

    Returns:
        solution: np.ndarray of shape (T,)
        solve_time: wall-clock solve time in seconds
        success: boolean indicating whether an optimal full solution was found
    """
    # --------------------------------------------------
    # Convert inputs
    # --------------------------------------------------
    obs = observations.detach().cpu().numpy().astype(int)
    trans = transition.detach().cpu().numpy()
    emit = emission.detach().cpu().numpy()
    init = initial.detach().cpu().numpy()

    T = int(obs.shape[0])
    K = int(trans.shape[0])

    if transition.shape[1] != K:
        raise ValueError("transition must have shape (K, K)")
    if emission.shape[0] != K:
        raise ValueError("emission must have shape (K, M)")
    if initial.shape[0] != K:
        raise ValueError("initial must have shape (K,)")

    if T <= 0:
        raise ValueError("Sequence length T must be positive")

    if N < 0:
        raise ValueError("N must be nonnegative")
    if N > 0 and N > T:
        raise ValueError(f"Subsequence length N={N} cannot exceed sequence length T={T}")
    if N > 0 and N > K:
        raise ValueError(f"Subsequence length N={N} cannot exceed number of states K={K}")

    # --------------------------------------------------
    # Convert to log-probs
    # --------------------------------------------------
    # NOTE:
    # This uses a large negative value for zero probabilities.
    # Shouldn't really matter, since we sample from a continuous distribution.
    NEG_INF = -1e12

    def safe_log(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, NEG_INF, dtype=float)
        mask = x > 0
        out[mask] = np.log(x[mask])
        return out

    log_init = safe_log(init)
    log_trans = safe_log(trans)
    log_emit = safe_log(emit)

    # Emission score by time/state
    emit_score = np.empty((T, K), dtype=float)
    for t in range(T):
        ot = obs[t]
        for k in range(K):
            emit_score[t, k] = log_emit[k, ot]

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    #Uses a flow-based formulation a la Koller-Friedman rather than Obben's formulation.
    #This is more efficient and creates less decision variables.
    
    model = ConcreteModel()

    model.T_IDX = RangeSet(0, T - 1)
    model.K_IDX = RangeSet(0, K - 1)

    if T > 1:
        model.T_ARC = RangeSet(0, T - 2)

    if N > 0:
        model.S_IDX = RangeSet(0, T - N)

    # Variables
    #z's are variable state trackers. z_{tk} is X_t = k.
    model.z = Var(model.T_IDX, model.K_IDX, domain=Binary)

    #y's are arc/transition trackers. y_{tjk} is X_{t-1} = j -> X_{t} = k
    if T > 1:
        model.y = Var(model.T_ARC, model.K_IDX, model.K_IDX, domain=Binary)

    #a's are the subsequence start trackers. a_{t} = 1 if [0,1,...,N-1] start at time t.
    if N > 0:
        model.a = Var(model.S_IDX, domain=Binary)

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------
    ### Standard HMM constraints
    # Exactly one state at each time
    def one_state_rule(m, t):
        return sum(m.z[t, k] for k in m.K_IDX) == 1

    model.one_state = Constraint(model.T_IDX, rule=one_state_rule)

    if T > 1:
        # Outflow: sum_j y[t,i,j] = z[t,i]
        # Number of active arcs out of X_t = k equals 0/1, whether X_t = k is True/False
        def outflow_rule(m, t, i):
            return sum(m.y[t, i, j] for j in m.K_IDX) == m.z[t, i]

        model.outflow = Constraint(model.T_ARC, model.K_IDX, rule=outflow_rule)

        # Inflow: sum_i y[t,i,j] = z[t+1,j]
        # Number of active arcs into X_t = k equals 0/1, whether X_t = k is True/False

        def inflow_rule(m, t, j):
            return sum(m.y[t, i, j] for i in m.K_IDX) == m.z[t + 1, j]

        model.inflow = Constraint(model.T_ARC, model.K_IDX, rule=inflow_rule)

    ### Additional inequalities from subsequence constraint
    if N > 0:
        # a[s] <= z[s+i, i] for all i=0,...,N-1
        def start_upper_rule(m, s, i):
            return m.a[s] <= m.z[s + i, i]

        model.start_upper = Constraint(range(T - N + 1), range(N), rule=start_upper_rule)

        # a[s] >= sum_i z[s+i, i] - N + 1
        def start_lower_rule(m, s):
            return m.a[s] >= sum(m.z[s + i, i] for i in range(N)) - N + 1

        model.start_lower = Constraint(model.S_IDX, rule=start_lower_rule)

        # At least one valid start
        def subseq_exists_rule(m):
            return sum(m.a[s] for s in m.S_IDX) >= 1

        model.subseq_exists = Constraint(rule=subseq_exists_rule)

    # --------------------------------------------------
    # Objective
    # --------------------------------------------------
    expr = 0.0

    # Initial + emission at t=0
    expr += sum(
        (log_init[k] + emit_score[0, k]) * model.z[0, k]
        for k in range(K)
    )

    # Emissions for t >= 1
    expr += sum(
        emit_score[t, k] * model.z[t, k]
        for t in range(1, T)
        for k in range(K)
    )

    # Transitions
    if T > 1:
        expr += sum(
            log_trans[i, j] * model.y[t, i, j]
            for t in range(T - 1)
            for i in range(K)
            for j in range(K)
        )

    model.obj = Objective(expr=expr, sense=maximize)

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    solver = SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create solver '{solver_name}'")

    if solver_name.lower() == "gurobi":
        if time_limit is not None:
            solver.options["TimeLimit"] = time_limit
        if mip_gap is not None:
            solver.options["MIPGap"] = mip_gap

    start_time = time.time()
    results = solver.solve(model, tee=tee)
    solve_time = time.time() - start_time

    # --------------------------------------------------
    # Check solver status
    # --------------------------------------------------
    status = results.solver.status
    term = results.solver.termination_condition

    acceptable =  (status == SolverStatus.ok) and (term == TerminationCondition.optimal)

    if not acceptable:
        print('Did not succeed in finding optimal solution')
        return [], -1, False
    # --------------------------------------------------
    # Extract solution
    # --------------------------------------------------
    solution = []
    for t in range(T):
        found = False
        for k in range(K):
            if value(model.z[t, k]) > 0.5:
                solution.append(k)
                found = True
                break
        if not found:
            print('Found solution did not provide a full assignment')
            return [], -1, False
        
    return solution, solve_time, True
