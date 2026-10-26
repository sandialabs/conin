"""The A* search shared by :func:`conin.inference.oracle.a_star` and
:func:`conin.inference.recursive.recursive_a_star`.

The search extends prefixes of hidden states (internal indices) one time step
at a time.  The cost of a prefix is its negative log-probability with the
observations; the heuristic is the cost of the best unconstrained completion,
so it is consistent and the solutions come out in order of cost.

The constraints are seen through a *search state*: ``initial(h)`` is the state
of the prefix ``(h,)`` and ``update(state, h)`` the state after appending
``h``.  ``feasible_partial(state, seq, T)`` prunes a prefix ``seq`` of a
sequence of length ``T`` and ``feasible(state, seq)`` accepts a complete
sequence.

With ``merge=True`` prefixes with the same ``(last state, length, search
state)`` lead to the same completions, so they form one search node.  The
search keeps the ``num_solutions`` cheapest prefixes of each node (a cheaper
prefix with the same node dominates every completion of a costlier one) and
expands each node at most ``num_solutions`` times, so the solutions are still
the ``num_solutions`` cheapest feasible sequences.  With ``merge=False``
(oracle constraints, whose state is the prefix itself) every prefix is its
own node, and the checks read the prefix itself (the state is ``None``).
"""

import heapq
import time

import numpy as np

from conin.inference.results import map_results, map_solution


class _HeapItem:
    """A heap entry, ordered by its priority only."""

    __slots__ = ("priority", "gscore", "seq", "state")

    def __init__(self, priority, gscore, seq, state):
        self.priority = priority
        self.gscore = gscore
        self.seq = seq
        self.state = state

    def __lt__(self, other):
        return self.priority < other.priority


def completion_costs(hmm, observed):
    """``V[t][h]``: the cost of the best unconstrained completion after ``h`` at ``t``.

    The cost is the negative log-probability of the later transitions and
    emissions (``inf`` if there is no completion with positive probability).
    """
    time_steps = len(observed)
    hidden_states = range(hmm.num_hidden_states)
    transition_mat = hmm.transition_mat.tolist()
    emission_mat = hmm.emission_mat.tolist()
    log_transition_mat = hmm.log_transition_mat.tolist()
    log_emission_mat = hmm.log_emission_mat.tolist()

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
    return V


def hmm_a_star(
    hmm,
    observed,
    *,
    initial=None,
    update=None,
    feasible_partial=None,
    feasible=None,
    merge=False,
    num_solutions=1,
    max_iterations=None,
    max_time=None,
    debug=False,
):
    """A* search for the ``num_solutions`` most likely feasible hidden sequences.

    Parameters
    ----------
    hmm : MatVecHMMRepresentation
        The HMM parameters.
    observed : list of int
        The internal indices of the observations (at least one).
    initial, update : callable, optional
        ``initial(h) -> state`` and ``update(state, h) -> state`` on internal
        hidden-state indices.  Without them the state is ``None``.
    feasible_partial : callable, optional
        ``feasible_partial(state, seq, T) -> bool`` for a prefix ``seq`` (a
        tuple of internal indices) of a sequence of length ``T``; it may
        accept prefixes that cannot be completed but must not reject one that
        can.  ``None`` accepts all.
    feasible : callable, optional
        ``feasible(state, seq) -> bool`` for a complete sequence.  ``None``
        accepts all.
    merge : bool, optional
        Merge prefixes with equal ``(last state, length, state)`` (the state
        must then be hashable).  See the module docstring.
    num_solutions, max_iterations, max_time, debug
        As for :func:`conin.inference.oracle.a_star`.

    Returns
    -------
    munch.Munch
        :func:`conin.inference.results.map_results` with ``states`` as tuples
        of internal indices and ``log_potential`` the search score
        (``log P(hidden, observed)``, a ``numpy.float64``).
    """
    start_time = time.time()
    time_steps = len(observed)

    hidden_states = range(hmm.num_hidden_states)
    start_vec = hmm.start_vec.tolist()
    transition_mat = hmm.transition_mat.tolist()
    emission_mat = hmm.emission_mat.tolist()
    log_start_vec = hmm.log_start_vec.tolist()
    log_transition_mat = hmm.log_transition_mat.tolist()
    log_emission_mat = hmm.log_emission_mat.tolist()

    V = completion_costs(hmm, observed)

    if merge:
        # node -> the negated scores of its best pushed prefixes (a max-heap of
        # at most num_solutions entries), and node -> number of expansions.
        best_pushed = {}
        num_popped = {}

    open_set = []

    def push(gscore, h, t, seq, state):
        # t is the length of seq; h its last state.
        if merge:
            node = (h, t, state)
            pushed = best_pushed.get(node)
            if pushed is None:
                best_pushed[node] = [-gscore]
            elif len(pushed) < num_solutions:
                heapq.heappush(pushed, -gscore)
            elif gscore < -pushed[0]:
                heapq.heapreplace(pushed, -gscore)
            else:
                # num_solutions prefixes of this node are at least as good.
                return
        item = _HeapItem(gscore + V[t - 1][h], gscore, seq, state)
        if t == 1:
            # The initial prefixes are heapified below.
            open_set.append(item)
        else:
            heapq.heappush(open_set, item)

    obs = observed[0]
    for h in hidden_states:
        if start_vec[h] > 0 and emission_mat[h][obs] > 0:
            state = initial(h) if initial is not None else None
            if feasible_partial is not None and not feasible_partial(
                state, (h,), time_steps
            ):
                continue
            push(-log_start_vec[h] - log_emission_mat[h][obs], h, 1, (h,), state)
    heapq.heapify(open_set)

    iteration = 0
    n_infeasible = 0
    termination_condition = "unknown"
    output = []
    while open_set:
        item = heapq.heappop(open_set)
        seq = item.seq
        state = item.state
        t = len(seq)
        h1 = seq[-1]

        if merge:
            node = (h1, t, state)
            count = num_popped.get(node, 0)
            if count >= num_solutions:
                # A costlier prefix of a node that was expanded enough.
                continue
            num_popped[node] = count + 1

        if t == time_steps:
            if feasible is None or feasible(state, seq):
                soln = map_solution(seq)
                soln.log_potential = np.float64(-item.priority)
                output.append(soln)
                if len(output) == num_solutions:
                    termination_condition = "ok"
                    break
            else:
                n_infeasible += 1

        else:
            gscore = item.gscore
            obs = observed[t]
            for h2 in hidden_states:
                if emission_mat[h2][obs] == 0.0 or transition_mat[h1][h2] == 0.0:
                    continue
                new_seq = seq + (h2,)
                new_state = update(state, h2) if update is not None else None
                if feasible_partial is not None and not feasible_partial(
                    new_state, new_seq, time_steps
                ):
                    continue
                push(
                    gscore - log_transition_mat[h1][h2] - log_emission_mat[h2][obs],
                    h2,
                    t + 1,
                    new_seq,
                    new_state,
                )

        iteration += 1
        if (max_iterations is not None) and (iteration >= max_iterations):
            termination_condition = f"max_iterations: {iteration}"
            break

        curr_time = time.time()
        if (max_time is not None) and ((curr_time - start_time) > max_time):
            termination_condition = f"max_time: {curr_time - start_time}"
            break

        if debug:
            if iteration % 100 == 0:
                print(f"  Iteration: {iteration}")
                print(f"  # Heap:    {len(open_set)}")
                print(f"  t:         {t}")
                print(f"  val:       {item.priority}")
                print(f"  ninfeas:   {n_infeasible}")
                print(f"  time:      {curr_time - start_time}")

    if termination_condition == "unknown":
        # The search space is exhausted.  (If max_iterations or max_time
        # stopped the search, termination_condition says so.)
        termination_condition = "ok" if output else "error: no feasible solutions"

    return map_results(
        output,
        solvetime=time.time() - start_time,
        termination_condition=termination_condition,
    )
