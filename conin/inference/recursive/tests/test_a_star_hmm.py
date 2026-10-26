"""Recursive A*: RecursiveConstraint input, regression tests and brute-force parity."""

import itertools

import pytest

import conin.examples.hidden_markov_model as tc
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.constraints import RecursiveConstraint
from conin.exceptions import InvalidInputError
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


def _check_solutions(cmodel, observed, results):
    """The first solution is the optimum; all are feasible and in order.

    Recursive A* merges prefixes with the same (last state, length, constraint
    state), so for ``num_solutions > 1`` it returns at most one sequence per
    merged class, not the k best sequences (a known limitation).
    """
    best = _brute_force(cmodel, observed, 1)
    assert results.solutions[0].log_likelihood == pytest.approx(best[0][0])
    lls = [s.log_likelihood for s in results.solutions]
    assert lls == sorted(lls, reverse=True)
    for sol in results.solutions:
        assert cmodel.is_feasible(sol.hidden)
        assert sol.log_likelihood == pytest.approx(
            cmodel.model.log_probability(observed, sol.hidden)
        )


def test_partial_feasibility_uses_the_new_state():
    # Regression: the extension was checked with
    # the parent's constraint state (and the parent's length t), so a prefix
    # that already broke the bound was kept.
    calls = []
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(0, 2, calls))
    observed = ["o0"] * 5
    results = recursive_a_star(hmm=cmodel, observed=observed)
    assert results.solutions[0].hidden.count("h0") <= 2
    # Every check after the first step sees the extended prefix's length and
    # a state that may exceed the bound (it is then pruned).
    assert max(n for n, t, T in calls) == 3
    assert {t for n, t, T in calls} == set(range(1, 6))


def test_tight_bound_prunes_at_once():
    # With ub=0 no h0 may appear: only the all-h1 path is ever extended.
    calls = []
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(0, 0, calls))
    results = recursive_a_star(hmm=cmodel, observed=["o0"] * 4)
    assert results.solutions[0].hidden == ["h1"] * 4
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
    assert all(s.hidden.count("h0") == 2 for s in results.solutions)
    _check_solutions(cmodel, observed, results)


@pytest.mark.parametrize("lb, ub", [(3, 5), (0, 2), (6, 8), (0, 8)])
@pytest.mark.parametrize("num_solutions", [1, 4])
def test_recursive_a_star_matches_brute_force(lb, ub, num_solutions):
    cmodel = tc.create_chmm1_recursive(lb, ub)
    observed = ["o1", "o0", "o0", "o1", "o1", "o0", "o0", "o1"]
    results = recursive_a_star(
        hmm=cmodel, observed=observed, num_solutions=num_solutions
    )
    _check_solutions(cmodel, observed, results)
    for sol in results.solutions:
        assert lb <= sol.hidden.count("h0") <= ub


def test_recursive_and_oracle_a_star_agree():
    observed = ["o0"] * 15
    rec = recursive_a_star(hmm=tc.create_chmm1_recursive(), observed=observed)
    ora = a_star(observed=observed, hmm=tc.create_chmm1_oracle())
    assert rec.solutions[0].hidden == ora.solution.hidden
    assert rec.solutions[0].log_likelihood == pytest.approx(ora.solution.log_likelihood)


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
    _check_solutions(cmodel, observed, results)
    for sol in results.solutions:
        assert 2 <= sol.hidden.count("h0") <= 3 and sol.hidden[-1] != "h0"


def test_bare_hmm_is_unconstrained_a_star():
    observed = ["o0", "o1", "o0"]
    rec = recursive_a_star(hmm=tc.create_hmm1(), observed=observed, num_solutions=3)
    ora = a_star(observed=observed, hmm=tc.create_hmm1(), num_solutions=3)
    assert [s.hidden for s in rec.solutions] == [s.hidden for s in ora.solutions]


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
    assert results == expected
    assert results.termination_condition == "ok"


def test_debug_prints_progress(capsys):
    cmodel = ConstrainedHiddenMarkovModel(tc.create_hmm1(), _num_h0(10, 12))
    recursive_a_star(hmm=cmodel, observed=["o1"] * 20, max_iterations=150, debug=True)
    assert "Iteration: 100" in capsys.readouterr().out
