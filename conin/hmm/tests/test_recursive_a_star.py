import pytest
import random

from conin.hmm.inference.recursive_a_star import *
from conin.hmm import *
from conin.hmm.inference import a_star
from conin import *


class Num_Zeros(HMMApplication):

    def __init__(self):
        self.num_zeros = None
        super().__init__(self.__class__.__name__)

    def initialize(
        self,
        *,
        hmm=None,
        prob_stay_in_same_state=None,
        prob_error=None,
        zero_start_prob=0.5,
        num_zeros,
        time
    ):
        if hmm is None:
            start_probs = {0: zero_start_prob, 1: 1 - zero_start_prob}
            transition_probs = {
                (0, 0): prob_stay_in_same_state,
                (0, 1): 1 - prob_stay_in_same_state,
                (1, 0): 1 - prob_stay_in_same_state,
                (1, 1): prob_stay_in_same_state,
            }
            emission_probs = {
                (0, 0): 1 - prob_error,
                (0, 1): prob_error,
                (1, 0): prob_error,
                (1, 1): 1 - prob_error,
            }
            hmm = HMM()
            hmm.load_model(
                start_probs=start_probs,
                transition_probs=transition_probs,
                emission_probs=emission_probs,
            )
            self.hmm = hmm
        else:
            self.hmm = hmm
        self.num_zeros = num_zeros
        self.generate_oracle_constraints()
        self._hidden_states = {0, 1}
        self._observable_states = {0, 1}
        self.time = time

    def run_simulations(
        self, *, num=1, debug=False, seed=None, with_observations=False
    ):
        if seed is not None:
            random.seed(seed)
        output = []
        for n in range(num):
            res = munch.Munch()
            hidden = self.oracle.generate_hidden(self.time)
            if with_observations:
                observed = self.oracle.generate_observed_from_hidden(hidden)
            res = munch.Munch(hidden=hidden, index=n)
            if with_observations:
                res.observed = observed
            output.append(res)
        return output
    
    def generate_oracle_constraints(self):
        constraint = has_exact_number_of_occurences_constraint(
            val=0, count=self.num_zeros
        )
        self.oracle.set_constraints([constraint])
    
    def initialize_constraint_data(self, hidden_state):
        if hidden_state == 0:
            return 1
        else:
            return 0
        
    def constraint_data_feasible_partial(self, *, constraint_data, t):
        return (constraint_data+(self.time-t) >= self.num_zeros) and constraint_data <= self.num_zeros
    
    def constraint_data_feasible(self, constraint_data):
        return constraint_data == self.num_zeros
    
    def update_constraint_data(self, *, hidden_state, constraint_data):
        if hidden_state == 0:
            return constraint_data + 1
        else:
            return constraint_data

@pytest.fixture
def app():
    prob_stay_in_same_state = 0.6  # 1/(1-prob_stay_in_same_state) = expected number of iterations of the same state
    prob_error = (
        0.3  # Proability that hidden state h has an observation which does not match it
    )
    num_zeros = 10  # Number of zeros
    time = 20
    app = Num_Zeros()
    app.initialize(
        prob_stay_in_same_state=prob_stay_in_same_state,
        prob_error=prob_error,
        num_zeros=num_zeros,
        time=time,
    )
    return app

@pytest.fixture
def heap_item():
    return Recursive_Heap_Item(
        priority=1, last_element="h0", length=10, constraint_data=(1, 2)
    )


class Test_Heap_Item:
    def test_init(self, heap_item):

        # Priority is a float
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority="a", last_element="h0", length=10, constraint_data=(1, 2)
            )

        # Last element is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element=["h0"], length=10, constraint_data=(1, 2)
            )

        # Length > 0
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=-10, constraint_data=(1, 2)
            )
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=0, constraint_data=(1, 2)
            )

        # constraint_data is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=10, constraint_data=[1, 2]
            )

    def test_getters(self, heap_item):
        assert heap_item.priority == 1
        assert heap_item.last_element == "h0"
        assert heap_item.length == 10
        assert heap_item.constraint_data == (1, 2)

    def test_setters(self, heap_item):
        # heap_item should be immutable
        with pytest.raises(AttributeError):
            heap_item.priority = 2
        with pytest.raises(AttributeError):
            heap_item.last_element = "h1"
        with pytest.raises(AttributeError):
            heap_item.length = 9
        with pytest.raises(AttributeError):
            heap_item.constraint_data = (2, 3)

    def test_lt(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h1", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=10, constraint_data=(1, 2)
        )

        assert not heap_item < x2
        assert heap_item < x3
        assert x2 < x3

    def test_eq(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=2, last_element="h1", length=10, constraint_data=(1, 2)
        )

        assert heap_item == x2
        assert not heap_item == x3
        assert not x2 == x3
        assert not heap_item == x4
        assert not x2 == x4
        assert not x3 == x4

    def test_hash(self, heap_item):
        hash(heap_item)

    def test_get_identifier(self, heap_item):
        assert heap_item.get_identifier() == ("h0", 10, (1, 2))


class Test_Unique_Heap:
    def test_init(self):
        A = Unique_Heapq()

    def test_add(self, heap_item):
        x2 = Recursive_Heap_Item(
            priority=1, last_element="h0", length=10, constraint_data=(1, 2)
        )
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=4, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=2, last_element="h1", length=10, constraint_data=(1, 2)
        )

        A = Unique_Heapq()
        A.add(heap_item)
        assert len(A) == 1
        A.add(x2)
        assert len(A) == 1
        A.add(x3)
        assert len(A) == 2
        A.add(heap_item)
        assert len(A) == 2
        A.add(x4)
        assert len(A) == 3
        A.add(x2)
        assert len(A) == 3
        A.add(x3)
        assert len(A) == 3
        A.add(x4)
        assert len(A) == 3
        assert set(A._Unique_Heapq__heap) == {x2, x3, x4}

    def test_pop(self, heap_item):
        x3 = Recursive_Heap_Item(
            priority=2, last_element="h0", length=4, constraint_data=(1, 2)
        )
        x4 = Recursive_Heap_Item(
            priority=3, last_element="h1", length=10, constraint_data=(1, 2)
        )

        A = Unique_Heapq()
        A.add(heap_item)
        A.add(x4)
        A.add(x3)
        while A:
            A.pop()
        assert len(A) == 0

        A.add(heap_item)
        A.add(x4)
        A.add(x3)

        x = A.pop()
        assert x == heap_item
        x = A.pop()
        assert x == x3
        x = A.pop()
        assert x == x4
        assert len(A) == 0

class Test_Inference:
    # These are copied a pasted from test_oracle_chmm.py, I should do some refactoring probably
    
    
    def test_a_star(self, app):
        observed = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(recursive_a_star(hmm_app=app, observed=observed))
        assert recursive_a_star(hmm_app=app, observed=observed).solutions[0].hidden == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        
    
    """
    def test_a_star_2(self, chmm):
        inference = Inference(statistical_model=chmm)
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]
        assert inference(observed).solutions[0].hidden == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    def test_a_star_mult(self, chmm):
        inference = Inference(statistical_model=chmm, num_solutions=2)
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]
        results = inference(observed)
        assert results.termination_condition == "ok"
        assert [sol.hidden for sol in results.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]

    def test_a_star_no_solution(self, chmm):
        inference = Inference(statistical_model=chmm)
        observed = ["o0"]
        results = inference(observed)
        assert results.termination_condition == "error: no feasible solutions"

    def test_a_star_not_enough_solutions(self, chmm):
        inference = Inference(statistical_model=chmm, num_solutions=2)
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]
        results = inference(observed)
        assert results.termination_condition == "ok"

    def test_a_star_deterministic_chmm(self):
        start_probs = {"h0": 1, "h1": 0}
        transition_probs = {
            ("h0", "h0"): 0,
            ("h0", "h1"): 1,
            ("h1", "h0"): 0,
            ("h1", "h1"): 1,
        }
        emission_probs = {
            ("h0", "o0"): 1,
            ("h0", "o1"): 0,
            ("h1", "o0"): 0,
            ("h1", "o1"): 1,
        }
        hmm = HMM()
        hmm.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
        hmm.set_seed(0)

        observed = ["o0", "o1", "o1", "o1"]
        inference = Inference(statistical_model=hmm)
        assert inference(observed).solutions[0].hidden == ["h0", "h1", "h1", "h1"]
    """