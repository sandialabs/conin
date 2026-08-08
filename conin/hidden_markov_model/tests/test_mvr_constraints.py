from itertools import product

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.hmm import HiddenMarkovModel
from conin.hidden_markov_model.mvr import HomMVR

from conin.hidden_markov_model.mvr_constraints import (
    mvr_constant,
    mvr_current_sequencelist,
    mvr_current_state,
    mvr_current_transition,
    mvr_forbid_sequencelist,
    mvr_forbid_state,
    mvr_forbid_transition,
    mvr_visit_sequencelist,
    mvr_visit_state,
    mvr_visit_transition,
)

from conin.hidden_markov_model.mvr_operators import (
    mvr_concatenate,
    mvr_count,
    mvr_not,
    mvr_precedence,
)

ALPHABET = ["a", "b", "c"]
MAX_LENGTH = 4


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_hmm(hidden_states=ALPHABET, observed_states=("o1", "o2")):
    """
    Build a uniform HiddenMarkovModel over the given hidden states.

    Only the hidden-state labels matter for these tests; the probabilities are
    uniform so that the model is valid.
    """
    hidden_states = list(hidden_states)
    observed_states = list(observed_states)

    num_hidden = len(hidden_states)
    num_observed = len(observed_states)

    hmm = HiddenMarkovModel()
    hmm.load_model(
        start_probs={h: 1.0 / num_hidden for h in hidden_states},
        transition_probs={
            (h1, h2): 1.0 / num_hidden for h1 in hidden_states for h2 in hidden_states
        },
        emission_probs={
            (h, o): 1.0 / num_observed for h in hidden_states for o in observed_states
        },
        initialize=True,
    )

    return hmm


def eval_mvr(mvr, seq):
    """
    Evaluate a HomMVR on a nonempty hidden-state sequence.
    """
    if len(seq) == 0:
        raise ValueError("MVR evaluation helper expects a nonempty sequence.")

    state = mvr.ini[seq[0]]

    for h in seq[1:]:
        state = mvr.upd[(state, h)]

    return mvr.evl[state]


def all_sequences(max_length=MAX_LENGTH, alphabet=ALPHABET):
    """
    Every nonempty sequence over `alphabet` of length up to `max_length`.
    """
    for length in range(1, max_length + 1):
        for seq in product(alphabet, repeat=length):
            yield list(seq)


def assert_matches(mvr, reference, max_length=MAX_LENGTH):
    """
    Check an MVR against a reference predicate by brute force.
    """
    for seq in all_sequences(max_length):
        assert eval_mvr(mvr, seq) is reference(seq), f"seq={seq!r}"


# ---------------------------------------------------------------------
# mvr_constant
# ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_constant_matches_reference(value):
    mvr = mvr_constant(make_hmm(), value)

    assert len(mvr.mediation_states) == 1
    assert_matches(mvr, lambda seq: value)


def test_constant_rejects_non_bool_value():
    with pytest.raises(InvalidInputError, match="value must be a bool"):
        mvr_constant(make_hmm(), 1)


# ---------------------------------------------------------------------
# mvr_current_state
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "states",
    [set(), {"a"}, {"b"}, {"a", "c"}, set(ALPHABET)],
)
def test_current_state_matches_reference(states):
    mvr = mvr_current_state(make_hmm(), states)
    assert_matches(mvr, lambda seq: seq[-1] in states)


@pytest.mark.parametrize("collection_type", [list, tuple, set, frozenset])
def test_current_state_accepts_any_collection_type(collection_type):
    mvr = mvr_current_state(make_hmm(), collection_type(["a"]))
    assert_matches(mvr, lambda seq: seq[-1] == "a")


def test_current_state_hidden_states_match_the_model():
    hmm = make_hmm()
    mvr = mvr_current_state(hmm, {"a"})
    assert mvr.hidden_states == hmm.hidden_states


@pytest.mark.parametrize(
    "states,match",
    [
        ({"z"}, "not hidden states"),
        # A bare label is ambiguous because strings are iterable.
        ("a", "must be a list, tuple, set"),
    ],
)
def test_current_state_rejects_invalid_input(states, match):
    with pytest.raises(InvalidInputError, match=match):
        mvr_current_state(make_hmm(), states)


# ---------------------------------------------------------------------
# mvr_current_transition
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "transitions",
    [
        set(),
        {("a", "b")},
        {("a", "a")},
        {("a", "b"), ("b", "c")},
        {(h1, h2) for h1 in ALPHABET for h2 in ALPHABET},
    ],
)
def test_transition_matches_reference(transitions):
    # The len >= 2 clause is what pins "no transition has been taken at t = 0".
    mvr = mvr_current_transition(make_hmm(), transitions)
    assert_matches(
        mvr,
        lambda seq: len(seq) >= 2 and (seq[-2], seq[-1]) in transitions,
    )


def test_transition_accepts_pairs_as_lists():
    mvr = mvr_current_transition(make_hmm(), [["a", "b"]])
    assert_matches(
        mvr,
        lambda seq: len(seq) >= 2 and (seq[-2], seq[-1]) == ("a", "b"),
    )


def test_transition_mediation_space_size():
    hmm = make_hmm()
    mvr = mvr_current_transition(hmm, {("a", "b")})
    assert len(mvr.mediation_states) == 2 * len(hmm.hidden_states)


@pytest.mark.parametrize(
    "transitions,match",
    [
        ({("a", "z")}, "not hidden states"),
        ({("a", "b", "c")}, "must be a \\(h_prev, h_curr\\) pair"),
        (("a", "b"), "must be a list, tuple, set"),
    ],
)
def test_transition_rejects_invalid_input(transitions, match):
    with pytest.raises(InvalidInputError, match=match):
        mvr_current_transition(make_hmm(), transitions)


# ---------------------------------------------------------------------
# mvr_current_sequencelist
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequences",
    [
        [],
        {("a",)},
        {("a", "b")},
        {("a", "a")},
        {("a",), ("b", "c")},
        {("a", "b", "a")},
        {("a", "b"), ("a", "c")},
        # Run-length: no other primitive can express "k consecutive a".
        {("a", "a", "a")},
        # Longer than any sequence we enumerate, so it can never match.
        {("a", "b", "c", "a", "b")},
    ],
)
def test_sequencelist_matches_reference(sequences):
    mvr = mvr_current_sequencelist(make_hmm(), sequences)

    def reference(seq):
        return any(
            tuple(seq[len(seq) - len(p) :]) == p
            for p in sequences
            if len(p) <= len(seq)
        )

    assert_matches(mvr, reference)


@pytest.mark.parametrize("states", [set(), {"a"}, {"a", "c"}, set(ALPHABET)])
def test_sequencelist_generalizes_current_state(states):
    hmm = make_hmm()
    general = mvr_current_sequencelist(hmm, [(s,) for s in sorted(states)])
    specific = mvr_current_state(hmm, states)

    for seq in all_sequences():
        assert eval_mvr(general, seq) is eval_mvr(specific, seq), f"seq={seq!r}"


@pytest.mark.parametrize(
    "transitions",
    [set(), {("a", "b")}, {("a", "a")}, {("a", "b"), ("b", "c")}],
)
def test_sequencelist_generalizes_transition(transitions):
    hmm = make_hmm()
    general = mvr_current_sequencelist(hmm, transitions)
    specific = mvr_current_transition(hmm, transitions)

    for seq in all_sequences():
        assert eval_mvr(general, seq) is eval_mvr(specific, seq), f"seq={seq!r}"


@pytest.mark.parametrize(
    "sequences",
    [
        {("a", "b", "a"), ("b", "c")},
        # Shared prefix: (), ("a",), ("a", "b"), ("a", "c") -- "a" is not duplicated.
        {("a", "b"), ("a", "c")},
    ],
)
def test_sequencelist_mediation_space_is_prefix_trie(sequences):
    mvr = mvr_current_sequencelist(make_hmm(), sequences)

    prefixes = {p[:i] for p in sequences for i in range(1, len(p) + 1)}

    assert len(mvr.mediation_states) == 1 + len(prefixes)


@pytest.mark.parametrize(
    "sequences,match",
    [
        (("a", "b", "a"), "even when it holds a single sequence"),
        (["a"], "even when it holds a single sequence"),
        ("ab", "must be a list, tuple, set"),
        ([{"a", "b"}], "must be a tuple or list"),
        ([()], "mvr_constant"),
        ([("a", "z")], "not hidden states"),
    ],
)
def test_sequencelist_rejects_invalid_input(sequences, match):
    with pytest.raises(InvalidInputError, match=match):
        mvr_current_sequencelist(make_hmm(), sequences)


# ---------------------------------------------------------------------
# Shared model validation (_model_hidden_states)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        lambda hmm: mvr_constant(hmm, True),
        lambda hmm: mvr_current_state(hmm, {"a"}),
        lambda hmm: mvr_current_transition(hmm, {("a", "b")}),
        lambda hmm: mvr_current_sequencelist(hmm, [("a",)]),
    ],
)
def test_primitives_reject_an_unusable_model(builder):
    with pytest.raises(InvalidInputError, match="no hidden states"):
        builder(HiddenMarkovModel())

    with pytest.raises(InvalidInputError, match="required argument"):
        builder(None)


# ---------------------------------------------------------------------
# Composition with operators
#
# The derivations the primitive layer exists to support. Visit/forbid are
# covered by the wrapper tests below; these are the operators the wrappers do
# not use.
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition,reference",
    [
        ("2", lambda n: n == 2),
        (">=1", lambda n: n >= 1),
        (">=2", lambda n: n >= 2),
        (">=3", lambda n: n >= 3),
    ],
)
def test_count_of_current_state_counts_occurrences(condition, reference):
    mvr = mvr_count(mvr_current_state(make_hmm(), {"a"}), condition)
    assert_matches(mvr, lambda seq: reference(seq.count("a")))


@pytest.mark.parametrize("pattern", [("a", "b"), ("a", "a")])
@pytest.mark.parametrize("k", [1, 2])
def test_count_of_sequencelist_counts_non_overlapping_matches(k, pattern):
    # mvr_count restarts the MVR after each acceptance, so overlapping matches are
    # not counted: "aaa" ends on ("a","a") twice but counts once.
    mvr = mvr_count(mvr_current_sequencelist(make_hmm(), {pattern}), f">={k}")

    def reference(seq):
        count = i = 0

        while i + len(pattern) <= len(seq):
            if tuple(seq[i : i + len(pattern)]) == pattern:
                count += 1
                i += len(pattern)
            else:
                i += 1

        return count >= k

    assert_matches(mvr, reference)


def test_precedence_of_current_states_is_first_occurrence_order():
    hmm = make_hmm()
    mvr = mvr_precedence(
        [mvr_current_state(hmm, {"a"}), mvr_current_state(hmm, {"b"})],
        "<",
    )

    def reference(seq):
        first_a = seq.index("a") if "a" in seq else len(seq) + 1
        first_b = seq.index("b") if "b" in seq else len(seq) + 1
        return first_a < first_b

    assert_matches(mvr, reference)


def test_concatenate_of_visits_is_always_appears_before():
    hmm = make_hmm()

    # No "a" occurs after any "b".
    mvr = mvr_not(
        mvr_concatenate([mvr_visit_state(hmm, {"b"}), mvr_visit_state(hmm, {"a"})])
    )

    def reference(seq):
        return not any(seq[i] == "b" and "a" in seq[i + 1 :] for i in range(len(seq)))

    assert_matches(mvr, reference)


# ---------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------


@pytest.mark.parametrize("states", [set(), {"a"}, {"a", "c"}, set(ALPHABET)])
def test_visit_and_forbid_state_match_reference(states):
    hmm = make_hmm()

    assert_matches(mvr_visit_state(hmm, states), lambda seq: bool(states & set(seq)))
    assert_matches(mvr_forbid_state(hmm, states), lambda seq: not (states & set(seq)))


@pytest.mark.parametrize(
    "transitions",
    [set(), {("a", "b")}, {("a", "a")}, {("a", "b"), ("b", "c")}],
)
def test_visit_and_forbid_transition_match_reference(transitions):
    hmm = make_hmm()

    def takes(seq):
        return any((seq[i], seq[i + 1]) in transitions for i in range(len(seq) - 1))

    assert_matches(mvr_visit_transition(hmm, transitions), takes)
    assert_matches(mvr_forbid_transition(hmm, transitions), lambda seq: not takes(seq))


@pytest.mark.parametrize(
    "sequences",
    [
        [],
        {("a",)},
        {("a", "b")},
        {("a", "a")},
        {("a",), ("b", "c")},
        {("a", "b", "a")},
    ],
)
def test_visit_and_forbid_sequencelist_match_reference(sequences):
    hmm = make_hmm()

    def contains(seq):
        return any(
            tuple(seq[i : i + len(p)]) == p
            for p in sequences
            for i in range(len(seq) - len(p) + 1)
        )

    assert_matches(mvr_visit_sequencelist(hmm, sequences), contains)
    assert_matches(
        mvr_forbid_sequencelist(hmm, sequences), lambda seq: not contains(seq)
    )


def test_wrappers_have_no_unreachable_mediation_states():
    # MVROperator.__call__ prunes operator output. The lift pairs each primitive
    # mediation state with a "satisfied yet" flag, but that flag cannot be True at
    # an accepting state, so those contradictory pairs are dropped.
    hmm = make_hmm()
    sequences = {("a", "b", "a"), ("b", "c")}

    assert len(mvr_current_state(hmm, {"a"}).mediation_states) == 2
    assert len(mvr_visit_state(hmm, {"a"}).mediation_states) == 3
    assert len(mvr_forbid_state(hmm, {"a"}).mediation_states) == 3

    assert len(mvr_current_sequencelist(hmm, sequences).mediation_states) == 6
    assert len(mvr_visit_sequencelist(hmm, sequences).mediation_states) == 10


@pytest.mark.parametrize(
    "builder,argument",
    [
        (mvr_visit_state, {"z"}),
        (mvr_forbid_state, {"z"}),
        (mvr_visit_transition, {("a", "z")}),
        (mvr_forbid_transition, {("a", "z")}),
        (mvr_visit_sequencelist, [("a", "z")]),
        (mvr_forbid_sequencelist, [("a", "z")]),
    ],
)
def test_wrappers_delegate_validation_to_the_primitive(builder, argument):
    with pytest.raises(InvalidInputError, match="not hidden states"):
        builder(make_hmm(), argument)


# ---------------------------------------------------------------------
# Numeric representation and CHMM integration
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        lambda hmm: mvr_constant(hmm, True),
        lambda hmm: mvr_current_state(hmm, {"a"}),
        lambda hmm: mvr_current_transition(hmm, {("a", "b")}),
        lambda hmm: mvr_current_sequencelist(hmm, {("a", "b", "a")}),
    ],
)
def test_primitives_build_a_valid_numeric_representation(builder):
    hmm = make_hmm()
    mvr = builder(hmm)

    repn = mvr.repn

    assert repn.num_hidden_states == len(hmm.hidden_states)
    assert repn.ini_array.shape == (
        len(hmm.hidden_states),
        len(mvr.mediation_states),
    )
    # Deterministic and total: one-hot over the current mediation axis.
    assert np.allclose(repn.ini_array.sum(axis=1), 1)
    assert np.allclose(repn.upd_array.sum(axis=1), 1)


def test_primitives_are_accepted_as_mvr_chmm_constraints():
    from conin.constraint import mvr_constraint_fn
    from conin.hidden_markov_model import ConstrainedHiddenMarkovModel

    hmm = make_hmm()

    @mvr_constraint_fn(name="never_visits_a")
    def never_visits_a(hidden_markov_model):
        return mvr_forbid_state(hidden_markov_model, {"a"})

    chmm = ConstrainedHiddenMarkovModel(hmm=hmm, constraints=[never_visits_a])
    chmm.initialize_chmm()

    assert len(chmm.chmm.constraints) == 1
    assert isinstance(chmm.chmm.constraints[0], HomMVR)
