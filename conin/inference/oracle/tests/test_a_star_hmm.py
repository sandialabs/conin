"""A* with oracle constraints: the partial-feasibility pruning and the partial functions it relies on."""

import itertools

import pytest

import conin.examples.hidden_markov_model as tc
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import OracleConstraint
from conin.constraints.oracle_library import (
    occurs_at_least_once_in_time_frame_constraint,
    occurs_only_in_time_frame_constraint,
)
from conin.exceptions import InvalidInputError
from conin.inference.oracle import a_star


def _brute_force(cmodel, observed, num_solutions=1):
    """The best feasible sequences by enumeration (log P(hidden, observed))."""
    hmm = cmodel.model
    scored = []
    for seq in itertools.product(hmm.hidden_states, repeat=len(observed)):
        seq = list(seq)
        if not cmodel.is_feasible(seq):
            continue
        p = (
            hmm.log_probability(observed, seq)
            if _possible(hmm, observed, seq)
            else None
        )
        if p is not None:
            scored.append((p, seq))
    scored.sort(key=lambda x: -x[0])
    return scored[:num_solutions]


def _possible(hmm, observed, seq):
    probs = [
        hmm.get_start_probs()[seq[0]],
        hmm.get_emission_probs()[seq[0], observed[0]],
    ]
    for t in range(1, len(seq)):
        probs.append(hmm.get_transition_probs()[seq[t - 1], seq[t]])
        probs.append(hmm.get_emission_probs()[seq[t], observed[t]])
    return all(p > 0 for p in probs)


class CountingOracle:
    """Counts the calls of a predicate and of its partial function."""

    def __init__(self, func, partial_func):
        self.calls = []
        self.partial_calls = []

        def f(states):
            self.calls.append(dict(states))
            return func(states)

        def pf(T, states):
            self.partial_calls.append(dict(states))
            return partial_func(T, states)

        self.constraint = OracleConstraint(func=f, partial_func=pf)


def test_pruning_rejects_the_extended_prefix_not_its_parent():
    # Regression: a_star_ checked
    # partial_is_feasible(seq) on the parent instead of the extension newSeq,
    # and `or isinstance(hmm, HMM_MatVecRepn)` made the condition always true.
    # "no h0 at all": every extension by h0 must be cut at once.
    counter = CountingOracle(
        func=lambda states: "h0" not in states.values(),
        partial_func=lambda T, states: "h0" not in states.values(),
    )
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), counter.constraint)
    observed = ["o0"] * 6
    results = a_star(observed=observed, hmm=cmodel)
    assert results.termination_condition == "ok"
    assert results.solution.hidden == ["h1"] * 6

    # The extensions are checked (prefixes of length 1..T), not the parents.
    assert {len(s) for s in counter.partial_calls} == set(range(1, 7))
    # Only the all-h1 path is ever expanded, so only it reaches the leaf check.
    assert counter.calls == [{t: "h1" for t in range(6)}]


def test_pruning_reduces_the_search_without_changing_the_optimum():
    cmodel = tc.create_chmm1_oracle()
    observed = ["o0"] * 20
    expected = ["h1"] * 8 + ["h0"] * 12
    # Without the pruning (it used to never fire) this search needs 983
    # iterations; with it, 506.
    assert (
        a_star(observed=observed, hmm=cmodel, max_iterations=506).solution.hidden
        == expected
    )
    # One iteration fewer finds no solution. a_star then raises IndexError
    # (it reads output[0]); that is a known bug, not what this test checks.
    with pytest.raises(IndexError):
        a_star(observed=observed, hmm=cmodel, max_iterations=505)


@pytest.mark.parametrize("T", [11, 12, 13, 14])
@pytest.mark.parametrize("num_solutions", [1, 3])
def test_chmm1_oracle_matches_brute_force(T, num_solutions):
    cmodel = tc.create_chmm1_oracle()
    observed = (["o1", "o0", "o1"] * 5)[:T]
    results = a_star(observed=observed, hmm=cmodel, num_solutions=num_solutions)
    expected = _brute_force(cmodel, observed, num_solutions)
    assert [s.hidden for s in results.solutions] == [seq for _, seq in expected]
    for sol, (p, _) in zip(results.solutions, expected):
        assert sol.log_likelihood == pytest.approx(p)


@pytest.mark.parametrize(
    "constraint",
    [
        occurs_only_in_time_frame_constraint("h0", lower_t=2),
        occurs_only_in_time_frame_constraint("h0", lower_t=1, upper_t=4),
        occurs_at_least_once_in_time_frame_constraint("h1", lower_t=3),
        occurs_at_least_once_in_time_frame_constraint("h0", lower_t=1, upper_t=3),
    ],
    ids=[
        "only_lower",
        "only_lower_upper",
        "at_least_once_lower",
        "at_least_once_window",
    ],
)
def test_time_frame_constraints_match_brute_force(constraint):
    # Regression: with upper_t=None, the partial functions of
    # these two library constraints rejected completable prefixes (a prefix
    # ending in val, resp. a prefix without val yet), so the fixed pruning
    # would cut the optimum.
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), constraint)
    observed = ["o0", "o1", "o0", "o0", "o1", "o0"]
    results = a_star(observed=observed, hmm=cmodel, num_solutions=2)
    expected = _brute_force(cmodel, observed, 2)
    assert [s.hidden for s in results.solutions] == [seq for _, seq in expected]


def test_time_frame_partial_functions_accept_completable_prefixes():
    only = occurs_only_in_time_frame_constraint("a", lower_t=2)
    # "a" at the last index of a prefix of a longer sequence is allowed.
    assert only.partial_func(4, {0: "b", 1: "b", 2: "a"})
    assert not only.partial_func(4, {0: "a"})
    # ... but not at the last index of the full sequence.
    assert not only.partial_func(4, {0: "b", 1: "b", 2: "b", 3: "a"})

    at_least = occurs_at_least_once_in_time_frame_constraint("a", lower_t=1)
    assert at_least.partial_func(3, {0: "b", 1: "b"})
    assert not at_least.partial_func(3, {0: "b", 1: "b", 2: "b"})
    assert at_least.partial_func(3, {0: "b", 1: "b", 2: "a"})


def test_a_star_bare_and_empty_constrained_models_agree():
    observed = ["o0", "o1", "o0", "o0"]
    bare = a_star(observed=observed, hmm=tc.create_hmm1(), num_solutions=3)
    empty = a_star(
        observed=observed,
        hmm=ConstrainedHiddenMarkovModel(tc.create_hmm1()),
        num_solutions=3,
    )
    assert [s.hidden for s in bare.solutions] == [s.hidden for s in empty.solutions]


def test_a_star_rejects_families_without_an_oracle_representation():
    with pytest.raises(InvalidInputError):
        a_star(observed=["o0"] * 3, hmm=tc.create_chmm1_pyomo())
    with pytest.raises(InvalidInputError):
        a_star(observed=["o0"] * 3, hmm=tc.create_chmm1_recursive())
