import heapq
import time
import numpy as np
import munch


class Recursive_Heap_Item:
    __slots__ = ["_priority", "_last_element", "_length", "_constraint_data"]

    def __init__(self, *, priority, last_element, length, constraint_data):
        """
        Initializes the object

        Parameters:
        priority (float): Priority in queue
        last_element (in H): Last element in sequence (must be hashable)
        length (int): Length of sequence (>= 0)
        constraint_data (any): Data needed to check constraint feasibility (must be hashable)
        """
        # Priority is a number
        if not isinstance(priority, (int, float)):
            raise TypeError(f"priority in Recursive_Heap_item must be a float.")

        # last_element is hashable
        try:
            hash(last_element)
        except:
            raise TypeError(f"last_element in Recursive_Heap_Item must be hashable.")

        # length is a positive integer
        if not isinstance(length, int):
            raise TypeError(f"length in Recursive_Heap_Item must be a int.")
        else:
            if length <= 0:
                raise TypeError(f"length in Recursive_Heap_Item must be >= 0.")

        # constraint_data is hashable
        try:
            hash(constraint_data)
        except:
            raise TypeError(f"constraint_data must be hashable.")

        self._priority = priority
        self._last_element = last_element
        self._length = length
        self._constraint_data = constraint_data

    # Getters
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

    def __lt__(self, y):
        """
        Only depends on priority
        """
        return self.priority < y.priority

    def __eq__(self, y):
        """
        Depends on all elements (not just priority)
        """
        return (
            (self.priority == y.priority)
            and (self.last_element == y.last_element)
            and (self.length == y.length)
            and (self.constraint_data == y.constraint_data)
        )

    def __hash__(self):
        """
        Just put everything in a tuple and take the hash of that
        """
        return hash(
            (self.priority, self.last_element, self.length, self.constraint_data)
        )

    def get_identifier(self):
        """
        Returns everything but the priority in a tuple.
        Used in Unique_Heap
        """
        return (self.last_element, self.length, self.constraint_data)


class Unique_Heapq:
    """
    This is used in our tailored A* algorithm
    We consider to elements "equal" if their sequences end with the same
    value and the have the same data associated with them.
    """

    def __init__(self):
        self.__heap = []
        self.__unique_map = {}  # Maps identifier to Recursive_Heap_Item

    def add(self, item):
        """
        Adds an item only if it doesn't already exist
        """
        identifier = item.get_identifier()

        # Check if the element is already in the heap
        if identifier in self.__unique_map:
            existing_item = self.__unique_map[identifier]

            if item.priority < existing_item.priority:
                self.__heap.remove(existing_item)
                heapq.heapify(self.__heap)  # Re-heapify after removal
                heapq.heappush(self.__heap, item)
                self.__unique_map[identifier] = item
        else:
            # Add the identifier to the map and the item to the heap
            self.__unique_map[identifier] = item
            heapq.heappush(self.__heap, item)

    def pop(self):
        """
        Pops the item with the smallest priority
        """
        while self.__heap:
            item = heapq.heappop(self.__heap)

            # Update our unique_map
            identifier = item.get_identifier()
            del self.__unique_map[identifier]

            return item
        raise IndexError("pop from an empty heap")

    def __len__(self):
        """
        Length of the heapq
        """
        return len(self.__heap)


def recursive_a_star(
    *,
    hmm_app,
    observed,
    num_solutions=1,
    debug=False,
    max_iterations=None,
    max_time=None,
):
    """
    This is a tailored version of the A* algorithm -- copied and modified
    from conin.hmm.inference.viterbi.

    Paramters:
        hmm_app: HMM_Application
        observed: List of observed states
        num_solutions: Number of solutions we wish to find
        debug: Whether to have additional printing
        max_iterations: Maximum number of iterations
        max_time: Maximum time of algorithm
    """
    start_time = time.time()
    knowledge_state_times_output = []
    obj_vals = []

    # Initalize variables
    hmm = hmm_app.get_hmm()
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
            constraint_data = hmm_app.initialize_constraint_data(h)

            if hmm_app.constraint_data_feasible_partial(
                constraint_data=constraint_data, t=1, time_steps=time_steps
            ):
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
                if hmm_app.constraint_data_feasible(constraint_data):
                    output.append(munch.Munch(hidden=list(seq), log_likelihood=-val))
                    obj_vals.append(-val)
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
                    new_constraint_data = hmm_app.update_constraint_data(
                        hidden_state=h2, constraint_data=constraint_data
                    )
                    if hmm_app.constraint_data_feasible_partial(
                        constraint_data=constraint_data, t=t, time_steps=time_steps
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
                termination_condition = f"max_time: {curr_time-start_time}"
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
                    print(f"  time:      {curr_time-start_time}")

    if len(output) < num_solutions:
        if num_solutions == 1:
            termination_condition = "error: no feasible solutions"
        else:
            termination_condition = "ok"

    ans = munch.Munch(
        observations=observed,
        solutions=output,
        termination_condition=termination_condition,
        knowledge_state_times=knowledge_state_times_output,
        objective_values=obj_vals,
    )

    return ans
