import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR

from conin.hidden_markov_model.mvr_operators import (
    mvr_timerange,
    mvr_and,
    mvr_or,
    mvr_not,
    mvr_not_yet,
    mvr_already_satisfied,
    mvr_sattime,
    mvr_setdiff,
    mvr_concatenate,
    mvr_concatenate_prefix,
    mvr_kfold_product,
    mvr_kfold_product_prefix,
    mvr_kleene_closure,
    mvr_kleene_closure_prefix,
    mvr_reverse,
    mvr_precedence,
    mvr_count,
)

ALPHABET = ["a", "b"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def eval_mvr(mvr, seq):
    """
    Evaluate a HomMVR/InhomMVR on a nonempty hidden-state sequence.

    These tests primarily construct HomMVRs, but this helper also supports
    InhomMVRs for convenience.
    """
    if len(seq) == 0:
        raise ValueError("MVR evaluation helper expects a nonempty sequence.")

    if isinstance(mvr, HomMVR):
        state = mvr.ini[seq[0]]

        for h in seq[1:]:
            state = mvr.upd[(state, h)]

        return mvr.evl[state]

    # Inhomogeneous case
    if len(seq) > mvr.time_horizon:
        raise ValueError("sequence length exceeds inhomogeneous MVR time horizon.")

    state = mvr.ini[seq[0]]

    for t, h in enumerate(seq[1:], start=0):
        state = mvr.upd[t][(state, h)]

    return mvr.evl[len(seq) - 1][state]


def assert_language(mvr, expected):
    """
    Check an MVR against a dictionary mapping strings to expected booleans.
    """
    for word, value in expected.items():
        assert eval_mvr(mvr, list(word)) is value, f"word={word!r}"


def contains_symbol_mvr(symbol):
    """
    Language: all nonempty words containing `symbol`.
    """
    hidden_states = list(ALPHABET)
    mediation_states = [False, True]

    ini = {h: h == symbol for h in hidden_states}

    upd = {
        (seen_prev, h): seen_prev or h == symbol
        for seen_prev in mediation_states
        for h in hidden_states
    }

    evl = {
        False: False,
        True: True,
    }

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def ends_symbol_mvr(symbol):
    """
    Language: all nonempty words whose final symbol is `symbol`.
    """
    hidden_states = list(ALPHABET)
    mediation_states = [False, True]

    ini = {h: h == symbol for h in hidden_states}

    upd = {
        (last_was_symbol, h): h == symbol
        for last_was_symbol in mediation_states
        for h in hidden_states
    }

    evl = {
        False: False,
        True: True,
    }

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def exact_word_mvr(word):
    """
    Language: exactly the single nonempty word `word`.

    State k means the prefix of length k has been matched.
    DEAD means the input can no longer match exactly.
    """
    if len(word) == 0:
        raise ValueError("exact_word_mvr expects a nonempty word.")

    hidden_states = list(ALPHABET)
    dead = ("DEAD", word)
    mediation_states = list(range(len(word) + 1)) + [dead]

    ini = {}

    for h in hidden_states:
        if h == word[0]:
            ini[h] = 1
        else:
            ini[h] = dead

    upd = {}

    for state in mediation_states:
        for h in hidden_states:
            if state == dead:
                next_state = dead
            elif state < len(word) and h == word[state]:
                next_state = state + 1
            else:
                next_state = dead

            upd[(state, h)] = next_state

    evl = {state: state == len(word) for state in mediation_states}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def trivial_inhom_mvr(time_horizon):
    """
    Language: every word within the horizon. Carries a horizon, unlike the rest.
    """
    return InhomMVR(
        hidden_states=list(ALPHABET),
        mediation_states=[[0]] * (time_horizon + 1),
        ini={h: 0 for h in ALPHABET},
        upd=[{(0, h): 0 for h in ALPHABET} for _ in range(time_horizon)],
        evl=[{0: True}] * (time_horizon + 1),
    )


# ---------------------------------------------------------------------
# One test per decorated operator
# ---------------------------------------------------------------------


def test_mvr_and():
    contains_a = contains_symbol_mvr("a")
    contains_b = contains_symbol_mvr("b")

    with pytest.warns(UserWarning, match="AND operator"):
        result = mvr_and([contains_a, contains_b])

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "b": False,
            "aa": False,
            "bb": False,
            "ab": True,
            "ba": True,
            "aba": True,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_and([])


def test_mvr_or():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_or([single_a, single_b])

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": True,
            "b": True,
            "aa": False,
            "ab": False,
            "ba": False,
            "bb": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_or([])


def test_mvr_not():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_not(single_a)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "b": True,
            "aa": True,
            "ab": True,
            "ba": True,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_not([single_a, single_b])


def test_mvr_not_yet():
    contains_a = contains_symbol_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_not_yet(contains_a)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "b": True,
            "bb": True,
            "bbb": True,
            "a": False,
            "ba": False,
            "ab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_not_yet([contains_a, single_b])


def test_mvr_already_satisfied():
    ends_a = ends_symbol_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_already_satisfied(ends_a)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "b": False,
            "bb": False,
            "a": True,
            "ba": True,
            "ab": True,
            "bba": True,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_already_satisfied([ends_a, single_b])


def test_mvr_sattime():
    contains_a = contains_symbol_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_sattime(contains_a)

    assert isinstance(result, HomMVR)
    assert result.prefix is True

    assert_language(
        result,
        {
            "a": True,
            "ba": True,
            "bba": True,
            "b": False,
            "bb": False,
            "aa": False,
            "aba": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_sattime([contains_a, single_b])


def test_mvr_setdiff():
    contains_a = contains_symbol_mvr("a")
    ends_a = ends_symbol_mvr("a")

    result = mvr_setdiff([contains_a, ends_a])

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "ba": False,
            "aa": False,
            "ab": True,
            "aab": True,
            "bb": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_setdiff([contains_a])


def test_mvr_concatenate():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_concatenate([single_a, single_b])

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "b": False,
            "ab": True,
            "aa": False,
            "ba": False,
            "bb": False,
            "abb": False,
            "aab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_concatenate([single_a])


def test_mvr_concatenate_prefix():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_concatenate_prefix([single_a, single_b])

    assert isinstance(result, HomMVR)
    assert result.prefix is True

    assert_language(
        result,
        {
            "a": False,
            "b": False,
            "ab": True,
            "aa": False,
            "ba": False,
            "bb": False,
            "abb": False,
            "aab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_concatenate_prefix([single_a])


def test_mvr_kfold_product():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    # Positional operator argument test: k is passed positionally.
    result = mvr_kfold_product(single_a, 2)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "aa": True,
            "aaa": False,
            "b": False,
            "ab": False,
            "ba": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_kfold_product([single_a, single_b], 2)

    with pytest.raises(InvalidInputError):
        mvr_kfold_product(single_a, 0)


def test_mvr_kfold_product_prefix():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    # Positional operator argument test: k is passed positionally.
    result = mvr_kfold_product_prefix(single_a, 2)

    assert isinstance(result, HomMVR)
    assert result.prefix is True

    assert_language(
        result,
        {
            "a": False,
            "aa": True,
            "aaa": False,
            "b": False,
            "ab": False,
            "ba": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_kfold_product_prefix([single_a, single_b], 2)

    with pytest.raises(InvalidInputError):
        mvr_kfold_product_prefix(single_a, 0)


def test_mvr_kleene_closure():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_kleene_closure(single_a)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": True,
            "aa": True,
            "aaa": True,
            "b": False,
            "ab": False,
            "ba": False,
            "aab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_kleene_closure([single_a, single_b])


def test_mvr_kleene_closure_prefix():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    result = mvr_kleene_closure_prefix(single_a)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": True,
            "aa": True,
            "aaa": True,
            "b": False,
            "ab": False,
            "ba": False,
            "aab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_kleene_closure_prefix([single_a, single_b])


def test_mvr_reverse():
    exact_ab = exact_word_mvr("ab")
    single_a = exact_word_mvr("a")

    result = mvr_reverse(exact_ab)

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "ba": True,
            "ab": False,
            "a": False,
            "b": False,
            "bb": False,
            "aa": False,
            "bab": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_reverse([exact_ab, single_a])


def test_mvr_precedence():
    contains_a = contains_symbol_mvr("a")
    contains_b = contains_symbol_mvr("b")

    # Positional operator argument test: relation is passed positionally.
    result = mvr_precedence([contains_a, contains_b], "<")

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": True,
            "aa": True,
            "ab": True,
            "aab": True,
            "abb": True,
            "ba": False,
            "bba": False,
            "bb": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_precedence([contains_a], "<")

    with pytest.raises(InvalidInputError):
        mvr_precedence([contains_a, contains_b], "!=")


def test_mvr_count():
    single_a = exact_word_mvr("a")
    single_b = exact_word_mvr("b")

    # Positional operator argument test: condition is passed positionally.
    result = mvr_count(single_a, "2")

    assert isinstance(result, HomMVR)

    assert_language(
        result,
        {
            "a": False,
            "aa": True,
            "aaa": False,
            "b": False,
            "ab": False,
            "ba": False,
            "aba": False,
        },
    )

    keyword_result = mvr_count(single_a, condition=">=2")

    assert isinstance(keyword_result, HomMVR)

    assert_language(
        keyword_result,
        {
            "a": False,
            "aa": True,
            "aaa": True,
            "b": False,
            "ab": False,
            "ba": False,
        },
    )

    with pytest.raises(InvalidInputError):
        mvr_count([single_a, single_b], "2")

    with pytest.raises(InvalidInputError):
        mvr_count(single_a, "not a valid condition")


def test_mvr_timerange():
    contains_a = contains_symbol_mvr("a")

    time_range = [1, 3]
    result = mvr_timerange(contains_a, time_range)

    assert result is contains_a  # inplace=True mutates and returns the input
    assert result.time_range == [1, 3]
    assert result.time_range is not time_range  # the MVR owns its range

    assert mvr_timerange(contains_a).time_range is None  # a bare call clears it

    # The horizon check applies on the inplace path, which skips the constructor.
    inhom = trivial_inhom_mvr(time_horizon=3)

    assert mvr_timerange(inhom, [5, 7]).time_range == [5, 7]

    with pytest.raises(InvalidInputError):
        mvr_timerange(inhom, [0, 4])

    # inplace=False rebuilds instead, leaving the input alone.
    single_a = exact_word_mvr("a")
    single_a.name = "single_a"
    single_a._prefix = True

    rebuilt = mvr_timerange(single_a, [1, 3], inplace=False)

    assert rebuilt is not single_a
    assert single_a.time_range is None

    assert rebuilt.time_range == [1, 3]
    assert rebuilt.name == "single_a"
    assert rebuilt._prefix is True

    assert_language(rebuilt, {"a": True, "b": False, "aa": False, "ab": False})
