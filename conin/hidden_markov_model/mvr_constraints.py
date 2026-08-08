from __future__ import annotations

from typing import Any

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR
from conin.hidden_markov_model.mvr_operators import mvr_already_satisfied, mvr_not_yet

# Bare labels are not accepted where a collection is expected: strings are
# iterable, and a label may itself be a sequence of labels, so allowing both
# forms would be ambiguous.
_COLLECTION_TYPES = (list, tuple, set, frozenset)


def _model_hidden_states(hidden_markov_model) -> list[Any]:
    """
    Validate and returns the external hidden-state labels of a loaded model.
    """
    if hidden_markov_model is None:
        raise InvalidInputError("hidden_markov_model is a required argument")

    hidden_states = getattr(hidden_markov_model, "hidden_states", None)

    if not hidden_states:
        raise InvalidInputError(
            "hidden_markov_model has no hidden states. "
            "Please run load_model() with the correct start/trans/emit probs"
        )

    return list(hidden_states)


# ------------------------------------------------------------------
# Base primitives
# ------------------------------------------------------------------


def mvr_constant(
    hidden_markov_model,
    value: bool,
) -> HomMVR:
    """
    The constantly true or constantly false MVR.
    """
    hidden_states = _model_hidden_states(hidden_markov_model)

    if not isinstance(value, bool):
        raise InvalidInputError("value must be a bool")

    only_state = ("__constant__",)
    mediation_states = [only_state]

    ini = {h: only_state for h in hidden_states}
    upd = {(only_state, h): only_state for h in hidden_states}
    evl = {only_state: value}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def mvr_current_state(
    hidden_markov_model,
    states,
) -> HomMVR:
    """
    MVR evaluates True iff the current hidden state is in "states".
    ie. If "states" = ['a'], at time t MVR will evaluate True iff hidden state is 'a'.
    """
    hidden_states = _model_hidden_states(hidden_markov_model)

    if not isinstance(states, _COLLECTION_TYPES):
        raise InvalidInputError(
            "states must be a list, tuple, set, or frozenset of hidden states, "
            "even when it holds a single state"
        )

    states = set(states)
    unknown = states - set(hidden_states)

    if unknown:
        raise InvalidInputError(
            "states contains labels that are not hidden states of "
            f"hidden_markov_model: {sorted(unknown, key=repr)}"
        )

    # The mediation state is the truth value of the predicate on the hidden
    # state just consumed, so evl is the identity.
    mediation_states = [False, True]

    ini = {h: h in states for h in hidden_states}
    upd = {(m, h): h in states for m in mediation_states for h in hidden_states}
    evl = {m: m for m in mediation_states}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def mvr_current_transition(
    hidden_markov_model,
    transitions,
) -> HomMVR:
    """
    MVR evaluates True iff the last transition is in "transitions".
    ie. If "transitions" = [('a','b')], at time t MVR will evaluate True iff h_t = 'b' and h_{t-1} = 'a'
    At t = 0 no transition has been taken, so the MVR evaluates False.
    """
    hidden_states = _model_hidden_states(hidden_markov_model)
    hidden_space = set(hidden_states)

    if not isinstance(transitions, _COLLECTION_TYPES):
        raise InvalidInputError(
            "transitions must be a list, tuple, set, or frozenset of "
            "(h_prev, h_curr) hidden state pairs"
        )

    # A bare pair is itself a collection of length two, so without this it would
    # be read as two malformed entries.
    if (
        isinstance(transitions, (tuple, list))
        and len(transitions) == 2
        and all(h in hidden_space for h in transitions)
    ):
        raise InvalidInputError(
            "transitions must be a list, tuple, set, or frozenset of "
            "(h_prev, h_curr) hidden state pairs, even when it holds a single "
            f"transition. Wrap it, e.g. [{tuple(transitions)!r}]"
        )

    transition_collection = set()

    for i, transition in enumerate(transitions):
        if not isinstance(transition, (tuple, list)) or len(transition) != 2:
            raise InvalidInputError(
                f"transitions entry {i} must be a (h_prev, h_curr) pair"
            )

        h_prev, h_curr = transition
        unknown = {h_prev, h_curr} - hidden_space

        if unknown:
            raise InvalidInputError(
                f"transitions entry {i} contains labels that are not hidden "
                f"states of hidden_markov_model: {sorted(unknown, key=repr)}"
            )

        transition_collection.add((h_prev, h_curr))

    # The mediation state pairs the hidden state just consumed -- needed as the
    # predecessor of the next transition -- with the truth value of the
    # predicate on the transition into it, so evl reads off the second entry.
    mediation_states = [(h, taken) for h in hidden_states for taken in [False, True]]

    # No transition has been taken when the first hidden state is consumed.
    ini = {h: (h, False) for h in hidden_states}

    upd = {
        ((h_prev, taken), h_curr): (
            h_curr,
            (h_prev, h_curr) in transition_collection,
        )
        for h_prev, taken in mediation_states
        for h_curr in hidden_states
    }

    evl = {(h, taken): taken for h, taken in mediation_states}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


def mvr_current_sequencelist(
    hidden_markov_model,
    sequences,
) -> HomMVR:
    """
    MVR evaluates True iff the hidden state sequence ends on one of "sequences".
    ie. If "sequences" = [('a','b','a')], at time t MVR will evaluate True iff
    h_{t-2}, h_{t-1}, h_t = 'a', 'b', 'a'.
    A sequence longer than t + 1 cannot have been completed, so the MVR evaluates False.

    This implements the Aho-Corasick construction for a more efficient MVR.
    """
    hidden_states = _model_hidden_states(hidden_markov_model)
    hidden_space = set(hidden_states)

    if not isinstance(sequences, _COLLECTION_TYPES):
        raise InvalidInputError(
            "sequences must be a list, tuple, set, or frozenset of hidden state "
            "sequences"
        )

    # A bare sequence is itself a collection of hidden states, so without this it
    # would be read as a list of malformed entries.
    if (
        isinstance(sequences, (tuple, list))
        and len(sequences) > 0
        and all(h in hidden_space for h in sequences)
    ):
        raise InvalidInputError(
            "sequences must be a list, tuple, set, or frozenset of hidden state "
            "sequences, even when it holds a single sequence. Wrap it, e.g. "
            f"[{tuple(sequences)!r}]"
        )

    patterns = set()

    for i, sequence in enumerate(sequences):
        if not isinstance(sequence, (tuple, list)):
            raise InvalidInputError(
                f"sequences entry {i} must be a tuple or list of hidden states"
            )

        if len(sequence) == 0:
            raise InvalidInputError(
                f"sequences entry {i} must be nonempty. Every sequence ends on the "
                "empty sequence, so use mvr_constant(hidden_markov_model, True) for "
                "the constantly true MVR"
            )

        unknown = set(sequence) - hidden_space

        if unknown:
            raise InvalidInputError(
                f"sequences entry {i} contains labels that are not hidden states "
                f"of hidden_markov_model: {sorted(unknown, key=repr)}"
            )

        patterns.add(tuple(sequence))

    # Aho-Corasick: the mediation state is the longest suffix of the hidden states
    # consumed so far that is a prefix of some sequence, so evl asks whether some
    # sequence is a suffix of that prefix.
    nodes = {()} | {
        pattern[:n] for pattern in patterns for n in range(1, len(pattern) + 1)
    }

    # Sorted so the mediation index order does not depend on set iteration order.
    mediation_states = sorted(nodes, key=lambda m: (len(m), tuple(map(repr, m))))

    upd = {}

    for m in mediation_states:
        for h in hidden_states:
            # Longest suffix of m + (h,) that is a node. Terminates because () is one.
            candidate = m + (h,)

            while candidate not in nodes:
                candidate = candidate[1:]

            upd[(m, h)] = candidate

    # The automaton starts at the root and consumes the first hidden state.
    ini = {h: upd[((), h)] for h in hidden_states}

    # When len(p) > len(m) the slice clamps to all of m, which cannot equal p.
    evl = {m: any(m[-len(p) :] == p for p in patterns) for m in mediation_states}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------
#
# Named compositions of the primitives above with the operators in
# mvr_operators.py. Nothing here builds ini/upd/evl by hand.
#
# The primitives are predicates on the current time step, so each is lifted over
# the whole chain by an operator: mvr_already_satisfied for "happens at least
# once", mvr_not_yet for "never happens". Validation is left to the primitive.


def mvr_visit_state(
    hidden_markov_model,
    states,
) -> HomMVR:
    """
    MVR evaluates True iff some hidden state up to time t is in "states".
    """
    return mvr_already_satisfied(mvr_current_state(hidden_markov_model, states))


def mvr_forbid_state(
    hidden_markov_model,
    states,
) -> HomMVR:
    """
    MVR evaluates True iff no hidden state up to time t is in "states".
    """
    return mvr_not_yet(mvr_current_state(hidden_markov_model, states))


def mvr_visit_transition(
    hidden_markov_model,
    transitions,
) -> HomMVR:
    """
    MVR evaluates True iff some transition up to time t is in "transitions".
    """
    return mvr_already_satisfied(
        mvr_current_transition(hidden_markov_model, transitions)
    )


def mvr_forbid_transition(
    hidden_markov_model,
    transitions,
) -> HomMVR:
    """
    MVR evaluates True iff no transition up to time t is in "transitions".
    """
    return mvr_not_yet(mvr_current_transition(hidden_markov_model, transitions))


def mvr_visit_sequencelist(
    hidden_markov_model,
    sequences,
) -> HomMVR:
    """
    MVR evaluates True iff the hidden states up to time t contain one of
    "sequences" as a contiguous subsequence.
    """
    return mvr_already_satisfied(
        mvr_current_sequencelist(hidden_markov_model, sequences)
    )


def mvr_forbid_sequencelist(
    hidden_markov_model,
    sequences,
) -> HomMVR:
    """
    MVR evaluates True iff the hidden states up to time t contain none of
    "sequences" as a contiguous subsequence.
    """
    return mvr_not_yet(mvr_current_sequencelist(hidden_markov_model, sequences))
