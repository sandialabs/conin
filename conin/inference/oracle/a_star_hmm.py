import heapq
import numpy as np
import time

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.results import map_results, map_solution, score_hmm_solutions

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
    oracle=None,
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
        observed (list): The internal indices of the observed states
        oracle (OracleHMMRepresentation, optional): The feasibility checks
            (``representation("oracle")`` of a constrained HMM). A prefix is
            extended by a state only if ``oracle.partial_is_feasible`` accepts
            the extended prefix, and a full sequence is a solution only if
            ``oracle.is_feasible`` accepts it.
        hmm (MatVecHMMRepresentation): The HMM parameters
            (``representation("matvec")`` of the HMM; ``oracle.hmm`` if omitted)

    Returns:
        munch.Munch: ``solution`` (``None`` if no solution was found),
        ``solutions``, ``termination_condition`` and ``solvetime`` (see
        :mod:`conin.inference.results`); the ``states`` of the solutions are
        internal indices and their ``log_potential`` is the search score.
    """
    start_time = time.time()

    # Initalize variables
    time_steps = len(observed)

    if oracle is not None and hmm is None:
        hmm = oracle.hmm

    # Read the arrays once: lists are faster to index in the loops below.
    hidden_states = range(hmm.num_hidden_states)
    start_vec = hmm.start_vec.tolist()
    transition_mat = hmm.transition_mat.tolist()
    emission_mat = hmm.emission_mat.tolist()
    log_start_vec = hmm.log_start_vec.tolist()
    log_transition_mat = hmm.log_transition_mat.tolist()
    log_emission_mat = hmm.log_emission_mat.tolist()

    # Precompute V[t][h] - The log-probability of the shortest path starting at time
    #       t in hidden state h
    V = [[0 for h in hidden_states] for t in range(time_steps)]
    for t in range(time_steps - 2, -1, -1):
        obs = observed[t + 1]
        for h1 in hidden_states:
            temp = np.inf
            for h2 in hidden_states:
                if (transition_mat[h1][h2] != 0) and (emission_mat[h2][obs] != 0):
                    temp = min(
                        temp,
                        V[t + 1][h2]
                        - log_transition_mat[h1][h2]
                        - log_emission_mat[h2][obs],
                    )
            V[t][h1] = temp

    # gScore - tuple hidden sequence -> negative log-probability
    #   Maps sequence of states already visited to negative log-probabilities
    gScore = dict()
    # openSet - heap of [value, seq] pairs, where 'value' is the negative
    # log-probability of sequence 'seq'
    openSet = []

    # Initialize the heap with the starting states
    for h in hidden_states:
        tempGScore = np.inf
        if oracle is not None and not oracle.partial_is_feasible(
            T=time_steps, seq=(h,)
        ):
            continue
        if (start_vec[h] > 0) and (emission_mat[h][observed[0]] > 0):
            tempGScore = -log_start_vec[h] - log_emission_mat[h][observed[0]]
            # Use tuple here b/c Python doesn't hash a list
            gScore[(h,)] = tempGScore
            openSet.append(HeapItem(priority=tempGScore + V[0][h], seq=(h,)))
    heapq.heapify(openSet)

    iteration = 0
    n_infeasible = 0
    termination_condition = "unknown"
    output = []
    while openSet:
        val, seq = heapq.heappop(openSet)
        t = len(seq)

        if t == time_steps:
            if oracle is None or oracle.is_feasible(seq):
                soln = map_solution(seq)
                # np.float64, as when the scores were computed with numpy scalars
                soln.log_potential = np.float64(-val)
                output.append(soln)
                if len(output) == num_solutions:
                    termination_condition = "ok"
                    break
            else:
                n_infeasible += 1

        else:
            h1 = seq[t - 1]
            currentGScore = gScore[seq]
            obs = observed[t]
            for h2 in hidden_states:
                if emission_mat[h2][obs] == 0.0 or transition_mat[h1][h2] == 0.0:
                    continue
                newSeq = seq + (h2,)
                if oracle is not None and not oracle.partial_is_feasible(
                    T=time_steps, seq=newSeq
                ):
                    continue
                tempGScore = (
                    currentGScore
                    - log_transition_mat[h1][h2]
                    - log_emission_mat[h2][obs]
                )
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

    if len(output) < num_solutions and termination_condition == "unknown":
        # The search space is exhausted.  (If max_iterations or max_time
        # stopped the search, termination_condition says so.)
        if num_solutions == 1:
            termination_condition = "error: no feasible solutions"
        else:
            termination_condition = "ok"

    return map_results(
        output,
        solvetime=time.time() - start_time,
        termination_condition=termination_condition,
    )


def a_star(
    *,
    observed,
    hmm,
    **kwargs,
):
    """A* search for the most likely hidden states of an (oracle-)constrained HMM.

    Parameters
    ----------
    observed : list
        Observed labels.
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model. Constraints must have an ``"oracle"`` representation (no
        constraints or ``"oracle"`` constraints); others raise
        ``InvalidInputError``.
    **kwargs
        ``num_solutions``, ``max_iterations``, ``max_time``, ``debug``.

    Returns
    -------
    munch.Munch
        ``solution``, ``solutions``, ``termination_condition`` and
        ``solvetime``; each solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``) and
        ``solver_objective`` (``None``).  See :mod:`conin.inference.results`.
        If no feasible sequence is found (the constraints exclude every
        sequence, or ``max_iterations``/``max_time`` stops the search first),
        ``solution`` is ``None`` and ``solutions`` holds what was found.
    """
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    # Raises InvalidInputError for constraint families without an "oracle"
    # representation (anything other than no constraints/"oracle").
    return _a_star(cmodel.representation("oracle"), observed=observed, **kwargs)


def _a_star(oracle, *, observed, **kwargs):
    """:func:`a_star` on an ``"oracle"`` representation (external labels)."""
    matvec = oracle.hmm
    hmm = oracle.model
    if not oracle.constraints:
        oracle = None

    observed_ = [hmm.observed_to_internal[o] for o in observed]
    ans_ = a_star_(observed=observed_, oracle=oracle, hmm=matvec, **kwargs)

    # Convert internal indices back to external labels
    solutions = [
        map_solution([hmm.hidden_to_external[h] for h in sol.states])
        for sol in ans_.solutions
    ]
    results = map_results(
        solutions,
        solvetime=ans_.solvetime,
        termination_condition=ans_.termination_condition,
    )
    return score_hmm_solutions(results, hmm, observed)
