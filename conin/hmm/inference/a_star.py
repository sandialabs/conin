import math
import heapq
import numpy as np
import munch
import time

from conin.hmm import HiddenMarkovModel, HMM
from conin.hmm import hmm_application

from dataclasses import dataclass, field
from typing import Any


# A data class that only allows comparisons w.r.t. the priority value
@dataclass(order=True)
class HeapItem:
    priority: float
    seq: Any = field(compare=False)

    def __iter__(self):
        yield self.priority
        yield self.seq


def a_star_(
    *,
    observed,
    chmm=None,
    hmm=None,
    num_solutions=1,
    max_iterations=None,
    max_time=None,
    debug=False,
):
    # TODO clean up some of the naming here
    """
    Performs the a_star algorithm to find the most likely hidden states base on observed
    This works for constrained hmms or hmm where we require multiple solutions

    Parameters:
        observed (list): The states we perform inference on

    Raises:
        InsufficientSolutionsError: Not enough solutions given by algorithm
    """
    start_time = time.time()

    # Initalize variables
    time_steps = len(observed)

    # if isinstance(hmm, hmm_application.HMMApplication):
    #    hmm.generate_oracle_constraints()

    if chmm:
        hmm = chmm.hmm

    # Precompute log probabilities for emission and transmission matrices
    log_emission_mat = {
        (h1, o): np.log(hmm.emission_mat[h1][o])
        for h1 in hmm.hidden_states
        for o in hmm.observed_states
        if hmm.emission_mat[h1][o] > 0
    }
    log_transition_mat = {
        (h1, h2): np.log(hmm.transition_mat[h1][h2])
        for h1 in hmm.hidden_states
        for h2 in hmm.hidden_states
        if hmm.transition_mat[h1][h2] > 0
    }

    # Precompute V[t][h] - The log-probability of the shortest path starting at time
    #       t in hidden state h
    V = [[0 for h in hmm.hidden_states] for t in range(time_steps)]
    for t in range(time_steps - 2, -1, -1):
        obs = observed[t + 1]
        for h1 in hmm.hidden_states:
            temp = np.inf
            for h2 in hmm.hidden_states:
                if (hmm.transition_mat[h1][h2] != 0) and (
                    hmm.emission_mat[h2][obs] != 0
                ):
                    temp = min(
                        temp,
                        V[t + 1][h2]
                        - log_transition_mat[h1, h2]
                        - log_emission_mat[h2, obs],
                    )
            V[t][h1] = temp

    # gScore - tuple hidden sequence -> negative log-probability
    #   Maps sequence of states already visited to negative log-probabilities
    gScore = dict()
    # openSet - heap of [value, seq] pairs, where 'value' is the negative
    # log-probability of sequence 'seq'
    openSet = []

    # Initialize the heap with the starting states
    for h in hmm.hidden_states:
        tempGScore = np.inf
        if (hmm.start_vec[h] > 0) and (hmm.emission_mat[h][observed[0]] > 0):
            tempGScore = -np.log(hmm.start_vec[h]) - log_emission_mat[h, observed[0]]
            # Use tuple here b/c Python doesn't hash a list
            gScore[(h,)] = tempGScore
            openSet.append(HeapItem(priority=tempGScore + V[0][h], seq=(h,)))
    heapq.heapify(openSet)

    iteration = 0
    n_infeasible = 0
    termination_condition = "unknown"
    output = []
    while True:
        val, seq = heapq.heappop(openSet)
        t = len(seq)

        if t == time_steps:
            if chmm is None or chmm.internal_constrained_hmm.is_feasible(seq):
                output.append(
                    munch.Munch(variable_value=seq, hidden=seq, log_likelihood=-val)
                )
                if len(output) == num_solutions:
                    termination_condition = "ok"
                    break
            else:
                n_infeasible += 1

        else:
            h1 = seq[t - 1]
            currentGScore = gScore[seq]
            obs = observed[t]
            for h2 in hmm.hidden_states:
                if (
                    hmm.emission_mat[h2][obs] == 0.0
                    or hmm.transition_mat[h1][h2] == 0.0
                ):
                    continue
                if (
                    chmm is None
                    or chmm.internal_constrained_hmm.partial_is_feasible(
                        T=time_steps, seq=seq
                    )
                ) or (isinstance(hmm, HMM)):
                    tempGScore = (
                        currentGScore
                        - log_transition_mat[h1, h2]
                        - log_emission_mat[h2, obs]
                    )
                    newSeq = seq + (h2,)
                    gScore[newSeq] = tempGScore
                    heapq.heappush(
                        openSet,
                        HeapItem(priority=tempGScore + V[t][h2], seq=newSeq),
                    )

        iteration += 1
        if (max_iterations is not None) and (iteration >= max_iterations):
            termination_condition = f"max_iterations: {iteration}"
            break

        curr_time = time.time()
        if (max_time is not None) and ((curr_time - start_time) > max_time):
            termination_condition = f"max_time: {curr_time - start_time}"
            break

        if openSet == []:
            break

        if debug:
            if iteration % 100 == 0:
                print(f"  Iteration: {iteration}")
                print(f"  # Heap:    {len(openSet)}")
                print(f"  t:         {t}")
                print(f"  val:       {val}")
                print(f"  ninfeas:   {n_infeasible}")
                print(f"  time:      {curr_time - start_time}")

    if len(output) < num_solutions:
        if num_solutions == 1:
            termination_condition = "error: no feasible solutions"
        else:
            termination_condition = "ok"

    ans = munch.Munch(
        observations=observed,
        solution=output[0],
        solutions=output,
        termination_condition=termination_condition,
    )
    return ans


def a_star(
    *,
    observed,
    hmm,
    **kwargs,
):
    if isinstance(hmm, HMM):
        return a_star_(observed=observed, hmm=hmm, **kwargs)

    if isinstance(hmm, HiddenMarkovModel):
        chmm = None
    else:  # CHMM
        chmm = hmm
        hmm = chmm.hmm

    observed_ = [hmm.observed_to_internal[o] for o in observed]
    hmm_ = hmm.hmm  # HMM instance associated with the HiddenMarkovModel
    ans_ = a_star_(observed=observed_, chmm=chmm, hmm=hmm_, **kwargs)

    # Convert internal indices back to external labels
    solutions = []
    for sol in ans_.solutions:
        hidden = [hmm.hidden_to_external[h] for h in sol.hidden]
        solutions.append(
            munch.Munch(
                variable_value=hidden, hidden=hidden, log_likelihood=sol.log_likelihood
            )
        )

    return munch.Munch(
        observations=observed,
        solution=solutions[0],
        solutions=solutions,
        termination_condition=ans_.termination_condition,
    )
