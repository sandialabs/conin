from __future__ import annotations

import warnings

from typing import Any

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR
from conin.hidden_markov_model.mvr_operators import mvr_already_satisfied, mvr_not_yet

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
    hidden_space = set(hidden_states)

    if not isinstance(states, _COLLECTION_TYPES):
        raise InvalidInputError(
            "states must be a list, tuple, set, or frozenset of hidden states, "
            "even when it holds a single state"
        )

    states = set(states)
    unknown = states - hidden_space

    if unknown:
        raise InvalidInputError(
            "states contains labels that are not hidden states of "
            f"hidden_markov_model: {sorted(unknown, key=repr)}"
        )

    # The mediation state is the predicate on the hidden state just consumed.
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

    # A bare pair would otherwise be read as two malformed entries.
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

    mediation_states = [(h, taken) for h in hidden_states for taken in [False, True]]

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

    # A bare sequence would otherwise be read as a list of malformed entries.
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

    # The mediation state is the longest suffix of the hidden states consumed so far
    # that is a prefix of some sequence.
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
# Common Constraint Classes
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Visit/Forbidden
# ------------------------------------------------------------------


def mvr_visit_state(
    hidden_markov_model,
    states,
) -> HomMVR:
    """
    MVR evaluates True iff chain has hit "states" at some time up to the current time.
    """
    return mvr_already_satisfied(mvr_current_state(hidden_markov_model, states))


def mvr_forbid_state(
    hidden_markov_model,
    states,
) -> HomMVR:
    """
    MVR evaluates True iff chain hasn't visited "states".
    """
    return mvr_not_yet(mvr_current_state(hidden_markov_model, states))


def mvr_visit_transition(
    hidden_markov_model,
    transitions,
) -> HomMVR:
    """
    MVR evaluates True iff at least one transition up to time t is in "transition".
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
    MVR evaluates True iff chain has hit at least one sequence in "sequences".
    "sequences" is a general set of finite sequences, with possibly varying length.
    """
    return mvr_already_satisfied(
        mvr_current_sequencelist(hidden_markov_model, sequences)
    )


def mvr_forbid_sequencelist(
    hidden_markov_model,
    sequences,
) -> HomMVR:
    """
    MVR evaluates True iff chain has never hit a sequence in "sequences".
    "sequences" is a general set of finite sequences, with possibly varying length.
    """
    return mvr_not_yet(mvr_current_sequencelist(hidden_markov_model, sequences))


# ------------------------------------------------------------------
# Holding Time
# ------------------------------------------------------------------


def mvr_holdingtime(
    hidden_markov_model,
    k: int,
    states=None,
) -> HomMVR:
    """
    MVR evaluates True iff the chain stays in each state in "states" for at least
    k time steps, with a trailing run as the sole exception. If "states" is None,
    it defaults to the entire hidden space.

    IMPORTANT: A trailing run is the final run in the hidden sequence.
    ie. "aaabb", the trailing run is "bb".

    Here are examples of where ignoring the trailing run comes up:

        1. k = 3, 'aa' evaluates True. Single run = trailing run.
        2. k = 3, 'ab' evaluates False. First run len('a')=1 < 3.
        3. k = 3, 'aaab' evaluates True. First run len('aaa") >= 3, trailing 'b' ignored.

    Warns when no run can ever end short, which makes the MVR constantly true:
    when "states" is empty, when k = 1, or when the model has a single hidden
    state. These are accepted rather than rejected, but are degenerate enough to
    be worth flagging.
    """
    hidden_states = _model_hidden_states(hidden_markov_model)
    hidden_space = set(hidden_states)

    if type(k) is not int:
        # bool is an int subclass, so isinstance would admit True/False.
        raise InvalidInputError("k must be an int")

    if k < 1:
        raise InvalidInputError("k must be at least 1")

    if states is None:
        target_states = set(hidden_states)
    else:
        if not isinstance(states, _COLLECTION_TYPES):
            raise InvalidInputError(
                "states must be a list, tuple, set, or frozenset of hidden states, "
                "even when it holds a single state"
            )

        target_states = set(states)
        unknown = target_states - hidden_space

        if unknown:
            raise InvalidInputError(
                "states contains labels that are not hidden states of "
                f"hidden_markov_model: {sorted(unknown, key=repr)}"
            )

    fail_state = ("__holdingtime_fail__",)

    def run_start(h):
        return (h, 1 if h in target_states else k)

    mediation_states = [
        (h, count)
        for h in hidden_states
        for count in (range(1, k + 1) if h in target_states else [k])
    ]

    # Checks if the constraint is trivial
    can_fail = bool(target_states) and k > 1 and len(hidden_states) > 1

    if can_fail:
        mediation_states.append(fail_state)
    else:
        if not target_states:
            reason = "states is empty, so no run is constrained"
        elif k == 1:
            reason = "k = 1, and every run has length >= 1"
        else:
            reason = (
                "hidden_markov_model has a single hidden state, so no run ever ends"
            )

        warnings.warn(
            f"mvr_holdingtime is the constantly true MVR here: {reason}. "
            "Use mvr_constant(hidden_markov_model, True) if that is what you meant.",
            UserWarning,
        )

    ini = {h: run_start(h) for h in hidden_states}

    upd = {}

    for m in mediation_states:
        for h_curr in hidden_states:
            if m == fail_state:
                upd[(m, h_curr)] = fail_state
                continue

            h_prev, count = m

            if h_curr == h_prev:
                upd[(m, h_curr)] = (h_prev, min(count + 1, k))
            elif count < k:
                upd[(m, h_curr)] = fail_state
            else:
                upd[(m, h_curr)] = run_start(h_curr)

    # exemption for the trailing run.
    evl = {m: m != fail_state for m in mediation_states}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini=ini,
        upd=upd,
        evl=evl,
    )
