"""Recursive A*: RecursiveConstraint input, regression tests and brute-force parity."""

import itertools
import random

import pytest

import conin.examples.hidden_markov_model as tc
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import RecursiveConstraint
from conin.exceptions import InvalidInputError
from conin.graphical_models import random_hmm
from conin.inference.oracle import a_star
from conin.inference.recursive import recursive_a_star


def _num_h0(lb, ub, calls=None):
    def feasible_partial(n, t, T):
        if calls is not None:
            calls.append((n, t, T))
        return n + (T - t) >= lb and n <= ub

    return RecursiveConstraint(
        initial=lambda h: int(h == "h0"),
        update=lambda n, h: n + int(h == "h0"),
        feasible=lambda n: lb <= n <= ub,
        feasible_partial=feasible_partial,
        name="num_h0",
    )


def _brute_force(cmodel, observed, num_solutions):
    hmm = cmodel.model
    scored = []
    for seq in itertools.product(hmm.hidden_states, repeat=len(observed)):
        if cmodel.is_feasible(list(seq)):
            scored.append((hmm.log_probability(observed, list(seq)), list(seq)))
    scored.sort(key=lambda x: -x[0])
    return scored[:num_solutions]


def _check_solutions(cmodel, observed, results, num_solutions):
    """The solutions are the num_solutions most likely feasible sequences."""
    expected = _brute_force(cmodel, observed, num_solutions)
    assert [s.log_potential for s in results.solutions] == pytest.approx(
        [p for p, _ in expected]
    )
    for sol in results.solutions:
        assert cmodel.is_feasible(sol.states)
        assert sol.log_potential == pytest.approx(
            cmodel.model.log_probability(observed, sol.states)
        )


def test_partial_feasibility_uses_the_new_state():
    # Regression: the extension was checked with
    # the parent's constraint state (and the parent's length t), so a prefix
    # that already broke the bound was kept.
    calls = []
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(0, 2, calls))
    observed = ["o0"] * 5
    results = recursive_a_star(hmm=cmodel, observed=observed)
    assert results.solutions[0].states.count("h0") <= 2
    # Every check after the first step sees the extended prefix's length and
    # a state that may exceed the bound (it is then pruned).
    assert max(n for n, t, T in calls) == 3
    assert {t for n, t, T in calls} == set(range(1, 6))


def test_tight_bound_prunes_at_once():
    # With ub=0 no h0 may appear: only the all-h1 path is ever extended.
    calls = []
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(0, 0, calls))
    results = recursive_a_star(hmm=cmodel, observed=["o0"] * 4)
    assert results.solutions[0].states == ["h1"] * 4
    # 2 initial checks + 2 extensions at each of the 3 later steps.
    assert len(calls) == 2 + 3 * 2


def test_infeasible_leaves_are_skipped():
    # Without a partial check, infeasible leaves reach the search frontier and
    # must be skipped, not returned. (n_infeasible counts them; it is only
    # shown in the debug output.)
    no_partial = RecursiveConstraint(
        initial=lambda h: int(h == "h0"),
        update=lambda n, h: n + int(h == "h0"),
        feasible=lambda n: n == 2,
    )
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), no_partial)
    observed = ["o0"] * 4
    results = recursive_a_star(hmm=cmodel, observed=observed, num_solutions=3)
    assert results.termination_condition == "ok"
    assert len(results.solutions) == 3
    assert all(s.states.count("h0") == 2 for s in results.solutions)
    _check_solutions(cmodel, observed, results, 3)


@pytest.mark.parametrize("lb, ub", [(3, 5), (0, 2), (6, 8), (0, 8)])
@pytest.mark.parametrize("num_solutions", [1, 4])
def test_recursive_a_star_matches_brute_force(lb, ub, num_solutions):
    cmodel = tc.create_chmm1_recursive(lb, ub)
    observed = ["o1", "o0", "o0", "o1", "o1", "o0", "o0", "o1"]
    results = recursive_a_star(
        hmm=cmodel, observed=observed, num_solutions=num_solutions
    )
    _check_solutions(cmodel, observed, results, num_solutions)
    for sol in results.solutions:
        assert lb <= sol.states.count("h0") <= ub


def _accept_all():
    return RecursiveConstraint(
        initial=lambda h: None, update=lambda s, h: None, feasible=lambda s: True
    )


def test_merged_prefixes_still_give_the_k_best():
    # Regression: prefixes with the same (last state, length, constraint
    # state) were merged into one search node that kept only the best prefix,
    # so with num_solutions > 1 a sequence through a merged node was lost:
    # here the 4th solution was h1 h0 h0 h0 h0 (-6.474) instead of
    # h1 h1 h1 h1 h0 (-6.001).
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _accept_all())
    observed = ["o0", "o1", "o0", "o1", "o0"]
    results = recursive_a_star(hmm=cmodel, observed=observed, num_solutions=4)
    oracle = a_star(hmm=tc.create_hmm1(), observed=observed, num_solutions=4)
    assert [s.states for s in results.solutions] == [s.states for s in oracle.solutions]
    assert results.solutions[3].states == ["h1", "h1", "h1", "h1", "h0"]
    _check_solutions(cmodel, observed, results, 4)


@pytest.mark.parametrize("num_solutions", [1, 2, 5, 20])
@pytest.mark.parametrize("seed", range(4))
def test_random_hmms_match_brute_force(seed, num_solutions):
    # Three hidden states, so many prefixes share a (last state, length,
    # count) node.
    hmm = random_hmm(
        hidden_states=["a", "b", "c"], observed_states=["x", "y"], seed=seed
    )
    rng = random.Random(seed)
    observed = [rng.choice(["x", "y"]) for _ in range(6)]
    at_most_two_a = RecursiveConstraint(
        initial=lambda h: int(h == "a"),
        update=lambda n, h: n + int(h == "a"),
        feasible=lambda n: n <= 2,
        feasible_partial=lambda n, t, T: n <= 2,
    )
    cmodel = ConstrainedHiddenMarkovModel(hmm, at_most_two_a)
    results = recursive_a_star(
        hmm=cmodel, observed=observed, num_solutions=num_solutions
    )
    assert len(results.solutions) == num_solutions
    _check_solutions(cmodel, observed, results, num_solutions)


def test_more_solutions_than_sequences():
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _accept_all())
    results = recursive_a_star(hmm=cmodel, observed=["o0"] * 3, num_solutions=10)
    assert len(results.solutions) == 8
    assert results.termination_condition == "ok"
    _check_solutions(cmodel, observed=["o0"] * 3, results=results, num_solutions=10)


def test_recursive_and_oracle_a_star_agree():
    observed = ["o0"] * 15
    rec = recursive_a_star(hmm=tc.create_chmm1_recursive(), observed=observed)
    ora = a_star(observed=observed, hmm=tc.create_chmm1_oracle())
    assert rec.solutions[0].states == ora.solution.states
    assert rec.solutions[0].log_potential == pytest.approx(ora.solution.log_potential)


def test_multiple_recursive_constraints_combine():
    no_h0_at_end = RecursiveConstraint(
        initial=lambda h: h,
        update=lambda last, h: h,
        feasible=lambda last: last != "h0",
    )
    cmodel = ConstrainedHiddenMarkovModel(
        tc.create_hmm1(), [_num_h0(2, 3), no_h0_at_end]
    )
    observed = ["o0"] * 6
    results = recursive_a_star(hmm=cmodel, observed=observed, num_solutions=2)
    _check_solutions(cmodel, observed, results, 2)
    for sol in results.solutions:
        assert 2 <= sol.states.count("h0") <= 3 and sol.states[-1] != "h0"


def test_bare_hmm_is_unconstrained_a_star():
    observed = ["o0", "o1", "o0"]
    rec = recursive_a_star(hmm=tc.create_hmm1(), observed=observed, num_solutions=3)
    ora = a_star(observed=observed, hmm=tc.create_hmm1(), num_solutions=3)
    assert [s.states for s in rec.solutions] == [s.states for s in ora.solutions]


def test_is_feasible_for_recursive_constraints():
    cmodel = tc.create_chmm1_recursive(2, 3)
    assert cmodel.is_feasible(["h0", "h0", "h1"])
    assert not cmodel.is_feasible(["h0", "h1", "h1"])
    assert not cmodel.is_feasible(["h0"] * 4)


def test_other_families_are_rejected():
    with pytest.raises(InvalidInputError):
        recursive_a_star(hmm=tc.create_chmm1_oracle(), observed=["o0"] * 3)


# ---------------------------------------------------------------------
# max_iterations, max_time and debug
# ---------------------------------------------------------------------


def test_max_iterations_stops_the_search():
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(10, 12))
    results = recursive_a_star(hmm=cmodel, observed=["o1"] * 20, max_iterations=3)
    assert results.solutions == []
    assert results.termination_condition == "max_iterations: 3"


def test_max_time_stops_the_search():
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(10, 12))
    results = recursive_a_star(hmm=cmodel, observed=["o1"] * 20, max_time=0)
    assert results.solutions == []
    assert results.termination_condition.startswith("max_time: ")


def test_limits_that_are_not_reached_do_not_change_the_result():
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(2, 3))
    observed = ["o0", "o1", "o1", "o0", "o1"]
    expected = recursive_a_star(hmm=cmodel, observed=observed)
    results = recursive_a_star(
        hmm=cmodel, observed=observed, max_iterations=10**6, max_time=10**6
    )
    assert results.solutions == expected.solutions
    assert results.termination_condition == "ok"


def test_debug_prints_progress(capsys):
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(10, 12))
    recursive_a_star(hmm=cmodel, observed=["o1"] * 20, max_iterations=150, debug=True)
    assert "Iteration: 100" in capsys.readouterr().out
