import munch
import pytest

from conin.algorithms.inference.hmm_results import (
    hidden_states_from_unrolled,
    key_solutions_by_time,
    unrolled_evidence,
)
from conin.algorithms.inference.query import _observed_time_range
from conin.algorithms.inference.query_checks import full_query_variables
from conin.exceptions import InvalidInputError


def test_key_solutions_by_time():
    solns = [munch.Munch(states=["a", "b"]), munch.Munch(states=["b", "b"])]
    results = key_solutions_by_time(munch.Munch(solution=solns[0], solutions=solns))
    assert results.solution.states == {0: "a", 1: "b"}
    assert "hidden" not in results.solution
    assert results.solutions[1].states == {0: "b", 1: "b"}


@pytest.mark.parametrize(
    "keyed_by_time,expected", [(False, ["b", "c"]), (True, {1: "b", 2: "c"})]
)
def test_hidden_states_from_unrolled(keyed_by_time, expected):
    # Every solution is converted (the or_topas solutions used to keep the
    # unrolled keys: only results.solution was converted).
    states = {("H", 0): "a", ("H", 1): "b", ("H", 2): "c", ("E", 1): "x"}
    solns = [munch.Munch(states=dict(states)), munch.Munch(states=dict(states))]
    results = munch.Munch(solution=solns[0], solutions=solns)
    hidden_states_from_unrolled(results, start=1, stop=2, keyed_by_time=keyed_by_time)
    assert results.solution.states == expected
    assert results.solutions[1].states == expected


@pytest.mark.parametrize("variables", [None, [], ["A", "B"], ("B", "A")])
def test_full_query_variables(variables):
    assert full_query_variables(variables, ("A", "B")) is None


def test_full_query_variables_rejects_a_partial_query():
    with pytest.raises(RuntimeError, match="method='variable_elimination'"):
        full_query_variables(["A"], ("A", "B"))
    # Naming a node the model doesn't have is a partial query too.
    with pytest.raises(RuntimeError, match="Partial MAP queries"):
        full_query_variables(["A", "C"], ("A", "B"))


def test_observed_time_range():
    observed = ["a", "b", "c"]
    assert _observed_time_range(observed, 0, None) == (0, 2)
    assert _observed_time_range(observed, 0, 1) == (0, 1)
    assert _observed_time_range(observed, 1, 2) == (1, 2)
    with pytest.raises(InvalidInputError, match="stop=3 is past the last"):
        _observed_time_range(observed, 0, 3)


def test_unrolled_evidence():
    assert unrolled_evidence(["a", "b", "c"], 1, 2) == {("E", 1): "b", ("E", 2): "c"}
