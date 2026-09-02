import warnings

import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.constrained_hmm import ConstrainedHiddenMarkovModel
from conin.hidden_markov_model.mvr_formula import build_mvr, build_mvr_functor

from conin.hidden_markov_model.mvr_constraints import (
    mvr_constant,
    mvr_current_sequencelist,
    mvr_current_state,
    mvr_current_transition,
    mvr_forbid_state,
    mvr_holdingtime,
    mvr_jump,
    mvr_jumpcounts,
    mvr_regex,
    mvr_visit_state,
    mvr_withinbounds,
)

from conin.hidden_markov_model.mvr_operators import (
    mvr_and,
    mvr_concatenate,
    mvr_count,
    mvr_kfold_product,
    mvr_kleene_closure,
    mvr_not,
    mvr_not_yet,
    mvr_or,
    mvr_precedence,
    mvr_reverse,
    mvr_sattime,
    mvr_setdiff,
)

from conin.hidden_markov_model.tests.test_mvr_constraints import (
    ALPHABET,
    skipif_no_greenery,
    MAX_LENGTH,
    NUMERIC_ALPHABET,
    all_sequences,
    eval_mvr,
    make_hmm,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def language(mvrs, alphabet=ALPHABET, max_length=MAX_LENGTH):
    """
    The language of a constraint list, which is the conjunction of its members.
    """
    if not isinstance(mvrs, list):
        mvrs = [mvrs]

    return {
        tuple(seq): all(eval_mvr(mvr, seq) for mvr in mvrs)
        for seq in all_sequences(max_length, alphabet)
    }


def assert_same_language(formula, expected, alphabet=ALPHABET, hmm=None):
    """
    Check a formula against the hand-composed calls it should lower to.
    """
    hmm = hmm if hmm is not None else make_hmm(hidden_states=alphabet)

    with warnings.catch_warnings():
        # mvr_and warns by design; several constructors warn when degenerate.
        warnings.simplefilter("ignore", UserWarning)
        parsed = build_mvr(hmm, formula)
        reference = expected(hmm)

    assert language(parsed, alphabet) == language(reference, alphabet)


def cs(hmm, *states):
    return mvr_current_state(hmm, set(states))


# ---------------------------------------------------------------------
# The grammar, against the calls each production lowers to.
#
# This table is the executable spec: one row per production, checked by brute
# force over every sequence up to MAX_LENGTH. test_grammar_covers_the_algebra
# below pins it against the public surface of both modules.
# ---------------------------------------------------------------------


FORMULAS = [
    # atoms
    ("a", lambda h: cs(h, "a")),
    ("<a>", lambda h: cs(h, "a")),
    ("(a, b)", lambda h: cs(h, "a", "b")),
    ("true", lambda h: mvr_constant(h, True)),
    ("false", lambda h: mvr_constant(h, False)),
    ("a -> b", lambda h: mvr_current_transition(h, [("a", "b")])),
    ("seq(a, b, c)", lambda h: mvr_current_sequencelist(h, [["a", "b", "c"]])),
    ("jump", lambda h: mvr_jump(h)),
    ("hold 2", lambda h: mvr_holdingtime(h, 2)),
    ("hold 2 in (a, b)", lambda h: mvr_holdingtime(h, 2, {"a", "b"})),
    ("hold 2 in a", lambda h: mvr_holdingtime(h, 2, {"a"})),
    ("jumps 2", lambda h: mvr_jumpcounts(h, "2")),
    ("jumps >= 2", lambda h: mvr_jumpcounts(h, ">=2")),
    ("jumps in [1,3]", lambda h: mvr_jumpcounts(h, "[1,3]")),
    # unary
    ("not a", lambda h: mvr_not(cs(h, "a"))),
    ("never a", lambda h: mvr_forbid_state(h, ["a"])),
    ("reach a", lambda h: mvr_visit_state(h, ["a"])),
    ("first a", lambda h: mvr_sattime(cs(h, "a"))),
    ("reverse a", lambda h: mvr_reverse(cs(h, "a"))),
    # counting
    ("count(a) 2", lambda h: mvr_count(cs(h, "a"), "2")),
    ("count(a) == 2", lambda h: mvr_count(cs(h, "a"), "2")),
    ("count(a) is 2", lambda h: mvr_count(cs(h, "a"), "2")),
    ("count(a) >= 2", lambda h: mvr_count(cs(h, "a"), ">=2")),
    ("count(a) < 3", lambda h: mvr_count(cs(h, "a"), "<3")),
    ("count(a) in [1,3]", lambda h: mvr_count(cs(h, "a"), "[1,3]")),
    ("count(a) in (1,3]", lambda h: mvr_count(cs(h, "a"), "(1,3]")),
    ("a[2]", lambda h: mvr_count(cs(h, "a"), "2")),
    # repetition
    ("a{2}", lambda h: mvr_kfold_product(cs(h, "a"), 2)),
    ("a+", lambda h: mvr_kleene_closure(cs(h, "a"))),
    # binary
    ("a cat b", lambda h: mvr_concatenate([cs(h, "a"), cs(h, "b")])),
    ("a but not b", lambda h: mvr_setdiff([cs(h, "a"), cs(h, "b")])),
    ("never a and reach b", lambda h: [mvr_forbid_state(h, ["a"]), mvr_visit_state(h, ["b"])]),
    ("never a or reach b", lambda h: mvr_or([mvr_forbid_state(h, ["a"]), mvr_visit_state(h, ["b"])])),
    # temporal: every relation
    ("reach a then reach b", lambda h: mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], "<")),
    ("reach a before reach b", lambda h: mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], "<")),
    ("reach a after reach b", lambda h: mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], ">")),
    ("reach a at or before reach b", lambda h: mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], "<=")),
    ("reach a at or after reach b", lambda h: mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], ">=")),
    # the issue's example
    ("reach b before a[3]", lambda h: mvr_precedence([mvr_visit_state(h, ["b"]), mvr_count(cs(h, "a"), "3")], "<")),
    # grouping: each row pins one precedence decision, so do not prune them
    ("a or b and c", lambda h: mvr_or([cs(h, "a"), mvr_and([cs(h, "b"), cs(h, "c")])])),
    ("never a cat b", lambda h: mvr_concatenate([mvr_forbid_state(h, ["a"]), cs(h, "b")])),
    ("a but not b cat c", lambda h: mvr_setdiff([cs(h, "a"), mvr_concatenate([cs(h, "b"), cs(h, "c")])])),
    ("never (a or b)", lambda h: mvr_not_yet(mvr_or([cs(h, "a"), cs(h, "b")]))),
    ("(a)", lambda h: cs(h, "a")),
    ("reach a then reach b then reach c", lambda h: mvr_and([
        mvr_precedence([mvr_visit_state(h, ["a"]), mvr_visit_state(h, ["b"])], "<"),
        mvr_precedence([mvr_visit_state(h, ["b"]), mvr_visit_state(h, ["c"])], "<")])),
]


@pytest.mark.parametrize("formula, expected", FORMULAS, ids=[f for f, _ in FORMULAS])
def test_formula_matches_hand_composed_calls(formula, expected):
    assert_same_language(formula, expected)


@skipif_no_greenery
@pytest.mark.parametrize("pattern", ["<a><b>*", "(<a>|<b>)+", "<a>{2,3}"])
def test_match_matches_hand_composed_calls(pattern):
    assert_same_language(f'match "{pattern}"', lambda h: mvr_regex(h, pattern))


def test_within_bounds_matches_hand_composed_calls():
    assert_same_language(
        "within [1, 2]",
        lambda h: mvr_withinbounds(h, [1, 2]),
        alphabet=NUMERIC_ALPHABET,
    )


def test_grammar_covers_the_algebra():
    """
    Deliberately redundant with the table above: it pins the coverage claim, so
    a new constructor or operator cannot be added without a formula reaching it.
    """
    from conin.hidden_markov_model import mvr_constraints, mvr_operators
    from conin.hidden_markov_model import mvr_formula as formula_module

    source = open(formula_module.__file__).read()
    public = {
        name
        for module in (mvr_constraints, mvr_operators)
        for name in vars(module)
        if name.startswith("mvr_") and not name.endswith("_fn")
    }
    # The visit/forbid family is not named by the grammar: a bare label lowers to
    # mvr_current_state, so "never"/"reach" reconstruct them compositionally.
    compositional = {
        "mvr_visit_state",
        "mvr_forbid_state",
        "mvr_visit_transition",
        "mvr_forbid_transition",
        "mvr_visit_sequencelist",
        "mvr_forbid_sequencelist",
    }

    missing = {name for name in public - compositional if name not in source}

    assert not missing, f"no formula production lowers to {sorted(missing)}"


# ---------------------------------------------------------------------
# time_range, the one thing every other operator drops
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula, windows",
    [
        ("reach b between 1 and 3", [[1, 3]]),
        ("reach b over [1,3]", [[1, 3]]),
        ("(never a cat reach b) between 1 and 3", [[1, 3]]),
        # binds tighter than "and", and a parenthesized window reaches each conjunct
        ("never a and reach b between 1 and 3", [None, [1, 3]]),
        ("(never a and reach b) between 1 and 3", [[1, 3], [1, 3]]),
        # mvr_timerange mutates by default, so a shared operand must survive
        ("reach b and reach b between 1 and 3", [None, [1, 3]]),
    ],
)
def test_window_attaches_a_time_range(formula, windows):
    assert [mvr.time_range for mvr in build_mvr(make_hmm(), formula)] == windows


# ---------------------------------------------------------------------
# Top-level conjunction lowers to a list, not a product
# ---------------------------------------------------------------------


def test_top_level_and_splits_without_building_a_product():
    hmm = make_hmm()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        parsed = build_mvr(hmm, "never a and reach b and reach c")

    assert len(parsed) == 3


def test_nested_and_builds_the_product():
    hmm = make_hmm()

    with pytest.warns(UserWarning, match="AND operator"):
        parsed = build_mvr(hmm, "not (never a and reach b)")

    assert len(parsed) == 1


# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula, match",
    [
        ("", "expected an expression"),
        ("never", "expected an expression"),
        ("a and", "expected an expression"),
        ("(a", r"expected '\)'"),
        ("a)", "unexpected"),
        ("a $ b", "unexpected character"),
        ("z", "not a hidden state"),
        ("seq(a, z)", "not a hidden state"),
        ("a[x]", "expected an integer"),
        ("a{2", r"expected '\}'"),
        ("count(a)", "expected a count condition"),
        ("count(a) in [1", "expected ','"),
        ("reach a then reach b after reach c", "one relation throughout"),
        ("match a", "quoted pattern"),
        ("reach b between 1", r"expected 'and'"),
        ("and a", "unexpected"),
        (None, "must be a string"),
    ],
)
def test_invalid_formula_is_rejected(formula, match):
    hmm = make_hmm()

    with pytest.raises(InvalidInputError, match=match):
        build_mvr(hmm, formula)


def test_error_points_at_the_offending_column():
    hmm = make_hmm()

    with pytest.raises(InvalidInputError) as excinfo:
        build_mvr(hmm, "never a and reach z")

    message = str(excinfo.value)

    assert "never a and reach z" in message
    assert "^" in message


def test_comparison_does_not_lex_as_a_quoted_label():
    # The label pattern is greedy, so a "<" once swallowed a later ">".
    assert len(build_mvr(make_hmm(), "count(a) < 2 and b -> c")) == 2


@pytest.mark.parametrize("label", ["count", "3", "A-1"])  # keyword, number, punctuation
def test_escaped_label_reaches_a_state_a_bare_label_cannot(label):
    alphabet = [label, "z1", "z2"]
    hmm = make_hmm(hidden_states=alphabet)

    assert language(build_mvr(hmm, f"never <{label}>"), alphabet) == language(
        mvr_forbid_state(hmm, [label]), alphabet
    )


@pytest.mark.parametrize("label", ["A B", "A<B", "A>B"])
def test_label_that_cannot_be_escaped_is_a_syntax_error(label):
    # Whitespace, "<" and ">" have no escape; they must fail loudly rather than
    # lex as something else.
    hmm = make_hmm(hidden_states=[label, "z1", "z2"])

    with pytest.raises(InvalidInputError, match="unexpected"):
        build_mvr(hmm, f"never <{label}>")


# ---------------------------------------------------------------------
# The deferred functor path
# ---------------------------------------------------------------------


def test_constraints_are_named_after_their_conjunct():
    # The name is what sat_time_mvr / sat_prob_mvr take as target="...".
    hmm = make_hmm()
    split = "never a and reach b"

    assert [m.name for m in build_mvr(hmm, split)] == [f"{split}[0]", f"{split}[1]"]
    assert build_mvr(hmm, "never a")[0].name == "never a"
    assert build_mvr(hmm, "never a", name="no_a")[0].name == "no_a"

    functor = build_mvr_functor("never a")[0]

    assert functor.name == "never a"
    assert functor(hmm).name == "never a"


def test_build_mvr_functor_reports_a_syntax_error_before_the_model_is_known():
    with pytest.raises(InvalidInputError, match="expected an expression"):
        build_mvr_functor("never")


def test_formula_drives_the_mvr_backend_end_to_end():
    hmm = make_hmm()
    chmm = ConstrainedHiddenMarkovModel(
        hmm=hmm, constraints=build_mvr_functor("never a and reach b")
    )
    chmm.initialize_chmm()

    assert chmm.constraint_type == "mvr"
    assert len(chmm.chmm.constraints) == 2
    assert language(list(chmm.chmm.constraints)) == language(
        [mvr_forbid_state(hmm, ["a"]), mvr_visit_state(hmm, ["b"])]
    )
