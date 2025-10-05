import math
import heapq
import numpy as np
import munch
import time

from conin.hmm import HiddenMarkovModel
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


def a_star(
    *,
    statistical_model,
    num_solutions=1,
    observed,
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
    hmm = statistical_model.get_hmm()
    internal_hmm = statistical_model.get_internal_hmm()
    time_steps = len(observed)
    internal_observed = [hmm.observed_to_internal[o] for o in observed]

    if isinstance(statistical_model, hmm_application.HMMApplication):
        statistical_model.generate_oracle_constraints()

    try:
        chmm = statistical_model.get_constrained_hmm()
    except BaseException:
        chmm = None

    # Precompute log probabilities for emission and transmission matrices
    log_emission_mat = {
        (h1, o): np.log(internal_hmm.emission_mat[h1][o])
        for h1 in internal_hmm.hidden_states
        for o in internal_hmm.observed_states
        if internal_hmm.emission_mat[h1][o] > 0
    }
    log_transition_mat = {
        (h1, h2): np.log(internal_hmm.transition_mat[h1][h2])
        for h1 in internal_hmm.hidden_states
        for h2 in internal_hmm.hidden_states
        if internal_hmm.transition_mat[h1][h2] > 0
    }

    # Precompute V[t][h] - The log-probability of the shortest path starting at time
    #       t in hidden state h
    V = [[0 for h in internal_hmm.hidden_states] for t in range(time_steps)]
    for t in range(time_steps - 2, -1, -1):
        obs = internal_observed[t + 1]
        for h1 in internal_hmm.hidden_states:
            temp = np.inf
            for h2 in internal_hmm.hidden_states:
                if (internal_hmm.transition_mat[h1][h2] != 0) and (
                    internal_hmm.emission_mat[h2][obs] != 0
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
    for h in internal_hmm.hidden_states:
        tempGScore = np.inf
        if (internal_hmm.start_vec[h] > 0) and (
            internal_hmm.emission_mat[h][internal_observed[0]] > 0
        ):
            tempGScore = (
                -np.log(hmm.start_vec[h]) - log_emission_mat[h, internal_observed[0]]
            )
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
                external_seq = [hmm.hidden_to_external[h] for h in seq]
                output.append(munch.Munch(hidden=external_seq, log_likelihood=-val))
                if len(output) == num_solutions:
                    termination_condition = "ok"
                    break
            else:
                n_infeasible += 1

        else:
            h1 = seq[t - 1]
            currentGScore = gScore[seq]
            obs = internal_observed[t]
            for h2 in internal_hmm.hidden_states:
                if (
                    internal_hmm.emission_mat[h2][obs] == 0.0
                    or internal_hmm.transition_mat[h1][h2] == 0.0
                ):
                    continue
                if (
                    chmm is None
                    or chmm.internal_constrained_hmm.partial_is_feasible(
                        T=time_steps, seq=seq
                    )
                ) or (isinstance(statistical_model, HiddenMarkovModel)):
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
        solutions=output,
        termination_condition=termination_condition,
    )
    return ans
