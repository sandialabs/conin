from __future__ import annotations

from typing import Any

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR

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


def mvr_transition(
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


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------
#
# Named compositions of the primitives above with the operators in
# mvr_operators.py. Nothing here builds ini/upd/evl by hand.
