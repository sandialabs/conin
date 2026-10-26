import heapq
import time
import numpy as np

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.results import map_results, map_solution, score_hmm_solutions


class Recursive_Heap_Item:
    __slots__ = [
        "_priority",
        "_last_element",
        "_length",
        "_constraint_data",
        "_valid",
        "_identifier",
    ]

    def __init__(self, *, priority, last_element, length, constraint_data):
        """
        Initializes the object

        Parameters:
        priority (float): Priority in queue
        last_element (in H): Last element in sequence (must be hashable)
        length (int): Length of sequence (>= 0)
        constraint_data (any): Data needed to check constraint feasibility (must be hashable)
        """
        if not isinstance(priority, (int, float)):
            raise TypeError("priority must be a number")
        if not isinstance(length, int) or length < 0:
            raise TypeError("length must be a non-negative integer")

        # Hash validation (only if needed - consider removing for performance)
        hash(last_element)
        hash(constraint_data)

        self._priority = priority
        self._last_element = last_element
        self._length = length
        self._constraint_data = constraint_data
        self._valid = True  # For lazy deletion
        self._identifier = None  # Cached identifier tuple

    # Properties using __slots__ are already optimized
    @property
    def priority(self):
        return self._priority

    @property
    def last_element(self):
        return self._last_element

    @property
    def length(self):
        return self._length

    @property
    def constraint_data(self):
        return self._constraint_data

    def __lt__(self, other):
        return self._priority < other._priority

    def __eq__(self, other):
        return (
            self._priority == other._priority
            and self._last_element == other._last_element
            and self._length == other._length
            and self._constraint_data == other._constraint_data
        )

    def __hash__(self):
        return hash(
            (
                self._priority,
                self._last_element,
                self._length,
                self._constraint_data,
            )
        )

    def get_identifier(self):
        """Returns cached identifier tuple (computed once)"""
        if self._identifier is None:
            self._identifier = (
                self._last_element,
                self._length,
                self._constraint_data,
            )
        return self._identifier

    def invalidate(self):
        """Mark this item as invalid for lazy deletion"""
        self._valid = False

    def is_valid(self):
        """Check if this item is still valid"""
        return self._valid


class Unique_Heapq:
    """
    Optimized version using lazy deletion to avoid expensive heap operations
    """

    def __init__(self):
        self.__heap = []
        self.__unique_map = {}  # Maps identifier to current best item

    def add(self, item):
        """Adds an item, replacing existing items with worse priority"""
        identifier = item.get_identifier()

        if identifier in self.__unique_map:
            existing_item = self.__unique_map[identifier]
            if item.priority < existing_item.priority:
                # Invalidate the old item instead of removing it
                existing_item.invalidate()
                self.__unique_map[identifier] = item
                heapq.heappush(self.__heap, item)
            # If new item has worse priority, ignore it
        else:
            self.__unique_map[identifier] = item
            heapq.heappush(self.__heap, item)

    def pop(self):
        """Pops the item with smallest priority, skipping invalidated items"""
        while self.__heap:
            item = heapq.heappop(self.__heap)
            if item.is_valid():
                identifier = item.get_identifier()
                del self.__unique_map[identifier]
                return item
            # Skip invalidated items
        raise IndexError("pop from an empty heap")

    def __len__(self):
        """Returns number of unique valid items"""
        return len(self.__unique_map)


def recursive_a_star(
    *,
    hmm,
    observed,
    num_solutions=1,
    debug=False,
    max_iterations=None,
    max_time=None,
):
    """A* search for the most likely hidden states under recursive constraints.

    A tailored version of :func:`conin.inference.oracle.a_star`: search nodes
    are ``(last hidden state, prefix length, constraint state)``, so prefixes
    with the same constraint state are merged.

    Parameters
    ----------
    hmm : ConstrainedHiddenMarkovModel or HiddenMarkovModel
        The model.  Its constraints must be :class:`~conin.constraints.RecursiveConstraint`
        objects (or none); other families raise ``InvalidInputError``.
    observed : list
        Observed labels.
    num_solutions : int, optional
        Number of solutions to find.
    debug : bool, optional
        Print progress every 100 iterations.
    max_iterations : int, optional
        Maximum number of iterations.
    max_time : float, optional
        Maximum time in seconds.

    Returns
    -------
    munch.Munch
        ``solution`` (``None`` if no solution was found), ``solutions``,
        ``termination_condition`` and ``solvetime``; each solution has
        ``states`` (the hidden states), ``log_potential``
        (``log P(hidden, observed)``) and ``solver_objective`` (``None``).
        See :mod:`conin.inference.results`.
    """
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    return _recursive_a_star(
        cmodel.representation("recursive"),
        observed=observed,
        num_solutions=num_solutions,
        debug=debug,
        max_iterations=max_iterations,
        max_time=max_time,
    )


def _recursive_a_star(
    constraint,
    *,
    observed,
    num_solutions=1,
    debug=False,
    max_iterations=None,
    max_time=None,
):
    """:func:`recursive_a_star` on a ``"recursive"`` representation."""
    start_time = time.time()

    # Initalize variables
    hmm = constraint.model
    time_steps = len(observed)

    start_probs = hmm.get_start_probs()
    emission_probs = hmm.get_emission_probs()
    transition_probs = hmm.get_transition_probs()
    hidden_states = hmm.get_hidden_states()

    # Precompute log probabilities for emission and transmission matrices
    log_emission_probs = {
        (h, o): np.log(emission_probs[(h, o)])
        for (h, o) in emission_probs.keys()
        if emission_probs[(h, o)] > 0
    }
    log_transition_probs = {
        (h1, h2): np.log(transition_probs[(h1, h2)])
        for (h1, h2) in transition_probs.keys()
        if transition_probs[(h1, h2)] > 0
    }

    # Precompute V[t][h] - The log-probability of the shortest path starting at time
    #       t in hidden state h
    V = {(t, h): 0 for t in range(time_steps) for h in hidden_states}

    for t in range(time_steps - 2, -1, -1):
        o = observed[t + 1]
        for h1 in hidden_states:
            temp = np.inf
            for h2 in hidden_states:
                if (transition_probs[(h1, h2)] != 0) and (emission_probs[(h2, o)] != 0):
                    temp = min(
                        temp,
                        V[(t + 1, h2)]
                        - log_transition_probs[(h1, h2)]
                        - log_emission_probs[(h2, o)],
                    )
            V[(t, h1)] = temp

    get_gScore = dict()  # Maps recursive heap item to negative log-probabilities
    get_seq = dict()  # Maps recursive heap item ids to sequences

    openSet = Unique_Heapq()

    # Initialize the heap with the starting states
    for h in hidden_states:
        if (
            (h in start_probs.keys())
            and (start_probs[h] > 0)
            and ((h, observed[0]) in emission_probs.keys())
            and (emission_probs[(h, observed[0])] > 0)
        ):
            gScore = -np.log(start_probs[h]) - log_emission_probs[(h, observed[0])]
            constraint_data = constraint.initial(h)

            if constraint.feasible_partial(constraint_data, 1, time_steps):
                item = Recursive_Heap_Item(
                    priority=gScore + V[(0, h)],
                    last_element=h,
                    length=1,
                    constraint_data=constraint_data,
                )

                get_gScore[item] = gScore
                openSet.add(item)
                get_seq[item] = (h,)

    iteration = 0
    n_infeasible = 0
    termination_condition = "unknown"
    output = []

    if len(openSet) == 0:
        termination_condition = "error: no feasible solutions"

    else:
        while True:
            item = openSet.pop()
            val = item.priority
            constraint_data = item.constraint_data
            t = item.length
            h1 = item.last_element

            gScore = get_gScore.pop(item)
            seq = get_seq.pop(item)

            if t == time_steps:
                if constraint.feasible(constraint_data):
                    output.append(map_solution(list(seq)))
                    if len(output) == num_solutions:
                        termination_condition = "ok"
                        break
                else:
                    n_infeasible += 1

            else:
                obs = observed[t]
                for h2 in hidden_states:
                    if (
                        emission_probs.get((h2, obs), 0.0) == 0.0
                        or transition_probs.get((h1, h2), 0.0) == 0.0
                    ):
                        continue
                    new_constraint_data = constraint.update(constraint_data, h2)
                    if constraint.feasible_partial(
                        new_constraint_data, t + 1, time_steps
                    ):
                        new_gScore = (
                            gScore
                            - log_transition_probs[(h1, h2)]
                            - log_emission_probs[(h2, obs)]
                        )

                        new_item = Recursive_Heap_Item(
                            priority=new_gScore + V[(t, h2)],
                            last_element=h2,
                            length=t + 1,
                            constraint_data=new_constraint_data,
                        )

                        get_gScore[new_item] = new_gScore
                        openSet.add(new_item)
                        get_seq[new_item] = seq + (h2,)

            iteration += 1
            if (max_iterations is not None) and (iteration >= max_iterations):
                termination_condition = f"max_iterations: {iteration}"
                break

            curr_time = time.time()
            if (max_time is not None) and ((curr_time - start_time) > max_time):
                termination_condition = f"max_time: {curr_time - start_time}"
                break

            if len(openSet) == 0:
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

    results = map_results(
        output,
        solvetime=time.time() - start_time,
        termination_condition=termination_condition,
    )
    return score_hmm_solutions(results, hmm, observed)
