"""Less-used option branches of the MVR operators: inhomogeneous inputs,
mixed horizons, fail-state renaming and the remaining count conditions."""

import warnings

from itertools import product

import pytest

from conin.constraints.mvr.base import HomMVR, InhomMVR
from conin.constraints.mvr.operators import (
    mvr_already_satisfied,
    mvr_and,
    mvr_count,
    mvr_not,
    mvr_not_yet,
    mvr_or,
    mvr_sattime,
)
from conin.constraints.mvr.tests.test_mvr_operators import (
    ALPHABET,
    contains_symbol_mvr,
    ends_symbol_mvr,
    eval_mvr,
    exact_word_mvr,
)

TIME_HORIZON = 3


def unroll(mvr, time_horizon=TIME_HORIZON):
    """The InhomMVR that behaves like ``mvr`` up to ``time_horizon``."""
    return InhomMVR(
        hidden_states=list(mvr.hidden_states),
        mediation_states=[list(mvr.mediation_states)] * (time_horizon + 1),
        ini=dict(mvr.ini),
        upd=[dict(mvr.upd)] * time_horizon,
        evl=[dict(mvr.evl)] * (time_horizon + 1),
    )


def words(max_length=TIME_HORIZON + 1):
    for length in range(1, max_length + 1):
        for word in product(ALPHABET, repeat=length):
            yield list(word)


def assert_same_language(left, right, max_length=TIME_HORIZON + 1):
    for word in words(max_length):
        assert eval_mvr(left, word) is eval_mvr(right, word), word


BASES = {
    "contains_a": lambda: contains_symbol_mvr("a"),
    "ends_b": lambda: ends_symbol_mvr("b"),
    "exact_ab": lambda: exact_word_mvr("ab"),
    "exact_a": lambda: exact_word_mvr("a"),
}

UNARY = {
    "not": mvr_not,
    "not_yet": mvr_not_yet,
    "already_satisfied": mvr_already_satisfied,
    "sattime": mvr_sattime,
    "count_2": lambda m: mvr_count(m, "2"),
    "count_le_1": lambda m: mvr_count(m, "<=1"),
    "count_lt_2": lambda m: mvr_count(m, "<2"),
    "count_gt_0": lambda m: mvr_count(m, ">0"),
    "count_ge_2": lambda m: mvr_count(m, ">=2"),
    "count_open_range": lambda m: mvr_count(m, "(0,2]"),
    "count_closed_range": lambda m: mvr_count(m, "[ 1 , 1 ]"),
}


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("op", UNARY)
def test_unary_operators_on_inhom_match_the_homogeneous_result(op, base):
    hom = BASES[base]()
    inhom_result = UNARY[op](unroll(hom))

    assert isinstance(inhom_result, InhomMVR)
    assert inhom_result.time_horizon == TIME_HORIZON
    assert_same_language(inhom_result, UNARY[op](hom))


def test_inhom_sattime_is_certified_prefix_free():
    result = mvr_sattime(unroll(contains_symbol_mvr("a")))

    assert result.prefix is True

    accepted = [tuple(w) for w in words() if eval_mvr(result, w)]
    assert accepted == [("a",), ("b", "a"), ("b", "b", "a"), ("b", "b", "b", "a")]


def test_inhom_count_restarts_after_each_acceptance():
    # Greedy, non-overlapping matches of "ab".
    result = mvr_count(unroll(exact_word_mvr("ab")), "2")

    for word in words():
        text = "".join(word)
        assert eval_mvr(result, word) is (text.count("ab") == 2), text


@pytest.mark.parametrize("inhom", [False, True])
def test_count_with_upper_zero_sends_initial_acceptance_to_fail(inhom):
    # A word starting with "a" is accepted at time 0, which already exceeds 0.
    base = contains_symbol_mvr("a")
    mvr = unroll(base) if inhom else base

    result = mvr_count(mvr, "0")

    for word in words():
        assert eval_mvr(result, word) is ("a" not in word), word


@pytest.mark.parametrize(
    "mixed",
    [
        lambda a, b: [unroll(a), b],
        lambda a, b: [a, unroll(b)],
        lambda a, b: [unroll(a), unroll(b)],
    ],
)
@pytest.mark.parametrize(
    "op,reducer",
    [(mvr_or, any), (mvr_and, all)],
)
@pytest.mark.filterwarnings("ignore:Use the AND operator")
def test_boolean_combination_of_mixed_inputs(mixed, op, reducer):
    a = contains_symbol_mvr("a")
    b = ends_symbol_mvr("b")

    result = op(mixed(a, b))

    assert isinstance(result, InhomMVR)
    assert result.time_horizon == TIME_HORIZON

    for word in words():
        expected = reducer([eval_mvr(a, word), eval_mvr(b, word)])
        assert eval_mvr(result, word) is expected, word


def test_boolean_combination_warns_on_unequal_horizons_and_uses_the_minimum():
    a = unroll(contains_symbol_mvr("a"), time_horizon=2)
    b = unroll(contains_symbol_mvr("b"), time_horizon=3)

    with pytest.warns(
        UserWarning,
        match="unequal horizons; using minimum horizon 2",
    ):
        result = mvr_or([a, b])

    assert result.time_horizon == 2


def test_boolean_combination_with_equal_horizons_does_not_warn():
    a = unroll(contains_symbol_mvr("a"))
    b = unroll(contains_symbol_mvr("b"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = mvr_or([a, b])

    assert result.time_horizon == TIME_HORIZON


@pytest.mark.parametrize("inhom", [False, True])
@pytest.mark.parametrize(
    "taken",
    [
        [("__sattime_fail__",)],
        [("__sattime_fail__",), ("__sattime_fail__", 0)],
    ],
)
def test_sattime_picks_a_fresh_fail_state(inhom, taken):
    # Contains-a, but with mediation states that collide with the fail label.
    no, yes = taken[0], "seen"
    states = [no, yes] + taken[1:]
    hom = HomMVR(
        hidden_states=list(ALPHABET),
        mediation_states=states,
        ini={h: yes if h == "a" else no for h in ALPHABET},
        upd={
            (m, h): yes if (m == yes or h == "a") else no
            for m in states
            for h in ALPHABET
        },
        evl={m: m == yes for m in states},
    )
    mvr = unroll(hom) if inhom else hom

    result = mvr_sattime(mvr)

    expected_fail = ("__sattime_fail__", len(taken) - 1)
    result_states = (
        set(result.mediation_states)
        if isinstance(result, HomMVR)
        else {m for states_t in result.mediation_states for m in states_t}
    )

    assert expected_fail in result_states
    assert result.prefix is True

    for word in words():
        first_a = word.index("a") if "a" in word else None
        assert eval_mvr(result, word) is (first_a == len(word) - 1), word
