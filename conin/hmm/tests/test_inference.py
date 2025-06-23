import pytest
import random

from conin.hmm.inference.recursive_a_star import *
from conin.hmm import *
from conin.hmm.inference import a_star
from conin import *
from conin.hmm.oracle_chmm import Oracle_CHMM
from conin.hmm.inference import ip_inference

import conin.hmm.tests.test_cases as tc

# TODO IP Testing
# TODO Viterbi/LP Testing


@pytest.fixture
def hmm():
    return tc.create_hmm1()


@pytest.fixture
def lb():
    return 10


@pytest.fixture
def ub():
    return 12


@pytest.fixture
def constraints(lb, ub):
    num_zeros_lb = has_minimum_number_of_occurences_constraint(val="h0", count=lb)
    num_zeros_ub = has_maximum_number_of_occurences_constraint(val="h0", count=ub)
    return [num_zeros_lb, num_zeros_ub]


@pytest.fixture
def chmm(hmm, constraints):

    chmm = Oracle_CHMM()
    chmm.load_model(hmm=hmm)
    for constraint in constraints:
        chmm.add_constraint(constraint)
    return chmm


@pytest.fixture
def recursive_app(hmm, constraints, lb, ub):
    app = tc.Num_Zeros()
    app.initialize(hmm=hmm, constraints=constraints, lb=lb, ub=ub)
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
                priority="a",
                last_element="h0",
                length=10,
                constraint_data=(1, 2),
            )

        # Last element is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1,
                last_element=["h0"],
                length=10,
                constraint_data=(1, 2),
            )

        # Length > 0
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1,
                last_element="h0",
                length=-10,
                constraint_data=(1, 2),
            )
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1, last_element="h0", length=0, constraint_data=(1, 2)
            )

        # constraint_data is hashable
        with pytest.raises(TypeError):
            x = Recursive_Heap_Item(
                priority=1,
                last_element="h0",
                length=10,
                constraint_data=[1, 2],
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


class Test_Inference_a_star:

    def test_a_star(self, chmm, recursive_app):
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]

        inference1 = Inference(statistical_model=chmm)
        inferred1 = inference1(observed).solutions[0].hidden

        inferred2 = (
            recursive_a_star(hmm_app=recursive_app, observed=observed)
            .solutions[0]
            .hidden
        )

        assert inferred1 == inferred2

        assert inferred2 == [
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

    def test_a_star_2(self, chmm, recursive_app):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]

        inference1 = Inference(statistical_model=chmm)
        inferred1 = inference1(observed).solutions[0].hidden

        inferred2 = (
            recursive_a_star(hmm_app=recursive_app, observed=observed)
            .solutions[0]
            .hidden
        )

        assert inferred1 == inferred2

        assert inferred2 == [
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

    def test_a_star_mult(self, chmm, recursive_app):
        observed = [
            "o1",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
        ]

        inference1 = Inference(statistical_model=chmm, num_solutions=2)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm_app=recursive_app, observed=observed, num_solutions=2
        )

        assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"
        assert [sol.hidden for sol in inferred1.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]
        assert [sol.hidden for sol in inferred2.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]

    def test_a_star_no_solution(self, chmm, recursive_app):
        observed = ["o0"]

        inference1 = Inference(statistical_model=chmm)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(hmm_app=recursive_app, observed=observed)

        assert inferred1.termination_condition == "error: no feasible solutions"

        assert inferred2.termination_condition == "error: no feasible solutions"

    def test_a_star_not_enough_solutions(self, chmm, recursive_app):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]
        inference1 = Inference(statistical_model=chmm, num_solutions=2)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm_app=recursive_app, observed=observed, num_solutions=2
        )
        assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"

    def test_a_star_deterministic_hmm(self):
        hmm = tc.create_hmm0()

        observed = ["o0", "o1", "o1", "o1"]
        inference = Inference(statistical_model=hmm)
        assert inference(observed).solutions[0].hidden == [
            "h0",
            "h1",
            "h1",
            "h1",
        ]


class Test_Inference_ip:

    def test_ip(self, recursive_app):
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]

        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].hidden

        inferred2 = (
            ip_inference(
                statistical_model=recursive_app,
                observed=observed,
                solver="glpk",
            )
            .solutions[0]
            .hidden
        )

        assert inferred2 == [
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

    def test_ip2(self, recursive_app):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]

        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].hidden

        inferred2 = (
            ip_inference(
                statistical_model=recursive_app,
                observed=observed,
                solver="glpk",
            )
            .solutions[0]
            .hidden
        )

        # assert inferred1 == inferred2

        assert inferred2 == [
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

    def Xtest_a_star_mult(self, chmm, recursive_app):
        observed = [
            "o1",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
        ]

        inference1 = Inference(statistical_model=chmm, num_solutions=2)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm_app=recursive_app, observed=observed, num_solutions=2
        )

        assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"
        assert [sol.hidden for sol in inferred1.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]
        assert [sol.hidden for sol in inferred2.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]

    def Xtest_a_star_no_solution(self, chmm, recursive_app):
        observed = ["o0"]

        inference1 = Inference(statistical_model=chmm)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(hmm_app=recursive_app, observed=observed)

        assert inferred1.termination_condition == "error: no feasible solutions"

        assert inferred2.termination_condition == "error: no feasible solutions"

    def Xtest_a_star_not_enough_solutions(self, chmm, recursive_app):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]
        inference1 = Inference(statistical_model=chmm, num_solutions=2)
        inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm_app=recursive_app, observed=observed, num_solutions=2
        )
        assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"

    def test_ip_deterministic_hmm(self):
        model = HMMApplication()
        model.hmm = tc.create_hmm0()

        observed = ["o0", "o1", "o1", "o1"]
        hidden = (
            ip_inference(statistical_model=model, observed=observed, solver="glpk")
            .solutions[0]
            .hidden
        )
        assert hidden == ["h0", "h1", "h1", "h1"]
