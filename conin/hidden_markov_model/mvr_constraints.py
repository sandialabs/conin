"""MVR constructors for a hidden Markov model.

This module is organized into two layers.

**Base primitives** are the MVRs that cannot be obtained by applying an operator
from :mod:`conin.hidden_markov_model.mvr_operators` to something simpler. Each
one is a predicate on a bounded window at the *end* of the hidden-state
sequence:

============================  ======  ==========================================
Primitive                     Window  Accepts iff
============================  ======  ==========================================
``mvr_constant``              0       the requested constant is ``True``
``mvr_current_state``         1       the last hidden state is in ``states``
``mvr_transition``            2       the last transition is in ``transitions``
============================  ======  ==========================================

Everything global or temporal is built on top of these with operators rather
than being constructed by hand. For example::

    visits_state    = mvr_already_satisfied(mvr_current_state(hmm, {"X"}))
    forbids_state   = mvr_not(visits_state)
    visits_X_twice  = mvr_count(mvr_current_state(hmm, {"X"}), "2")
    X_before_Y      = mvr_precedence(
                          [mvr_current_state(hmm, {"X"}),
                           mvr_current_state(hmm, {"Y"})],
                          "<",
                      )

**Convenience wrappers** name the compositions above so callers do not have to
spell them out. They belong in the second section of this module and are layered
strictly on top of the primitives -- they never construct ``ini``/``upd``/``evl``
themselves.

Notes
-----
The primitives deliberately do not accept a ``time_range``. Windowing is applied
to a finished MVR, and no operator propagates ``time_range`` yet, so a primitive
that carried one could not survive composition.
"""

from __future__ import annotations

from typing import Any

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import HomMVR

# A bare hidden state is not accepted where a collection of hidden states is
# expected. Strings are iterable, and a label may be both a hidden state and a
# sequence of hidden states, so accepting both forms would be ambiguous.
_COLLECTION_TYPES = (list, tuple, set, frozenset)


def _model_hidden_states(hidden_markov_model) -> list[Any]:
    """Return the external hidden-state labels of a loaded model.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
        Model whose hidden-state labels define the MVR alphabet.

    Returns
    -------
    list
        Hidden-state labels in internal index order.

    Raises
    ------
    InvalidInputError
        If the model is missing or has not been populated with ``load_model``.
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


def _validate_state_collection(
    states,
    hidden_states: list[Any],
    *,
    argument: str,
) -> set[Any]:
    """Validate a collection of hidden-state labels.

    Parameters
    ----------
    states : list, tuple, set, or frozenset
        Hidden-state labels to validate.
    hidden_states : list
        Hidden-state labels of the model.
    argument : str
        Argument name used in error messages.

    Returns
    -------
    set
        The validated labels.

    Raises
    ------
    InvalidInputError
        If ``states`` is not a supported collection or names a label that is not
        a hidden state of the model.
    """
    if not isinstance(states, _COLLECTION_TYPES):
        raise InvalidInputError(
            f"{argument} must be a list, tuple, set, or frozenset of hidden "
            "states, even when it holds a single state"
        )

    state_collection = set(states)
    unknown = state_collection - set(hidden_states)

    if unknown:
        raise InvalidInputError(
            f"{argument} contains labels that are not hidden states of "
            f"hidden_markov_model: {sorted(unknown, key=repr)}"
        )

    return state_collection


def _validate_transition_collection(
    transitions,
    hidden_states: list[Any],
    *,
    argument: str,
) -> set[tuple[Any, Any]]:
    """Validate a collection of ``(h_prev, h_curr)`` hidden-state pairs.

    Parameters
    ----------
    transitions : list, tuple, set, or frozenset
        Pairs of hidden-state labels to validate.
    hidden_states : list
        Hidden-state labels of the model.
    argument : str
        Argument name used in error messages.

    Returns
    -------
    set
        The validated pairs as 2-tuples.

    Raises
    ------
    InvalidInputError
        If ``transitions`` is not a supported collection, holds something other
        than a pair, or names a label that is not a hidden state of the model.
    """
    if not isinstance(transitions, _COLLECTION_TYPES):
        raise InvalidInputError(
            f"{argument} must be a list, tuple, set, or frozenset of "
            "(h_prev, h_curr) hidden state pairs"
        )

    hidden_space = set(hidden_states)

    # A bare (h_prev, h_curr) pair is itself a collection of length two, so
    # without this it would be read as two malformed entries.
    if (
        isinstance(transitions, (tuple, list))
        and len(transitions) == 2
        and all(h in hidden_space for h in transitions)
    ):
        raise InvalidInputError(
            f"{argument} must be a list, tuple, set, or frozenset of "
            "(h_prev, h_curr) hidden state pairs, even when it holds a single "
            f"transition. Wrap it, e.g. [{tuple(transitions)!r}]"
        )

    transition_collection = set()

    for i, transition in enumerate(transitions):
        if not isinstance(transition, (tuple, list)) or len(transition) != 2:
            raise InvalidInputError(
                f"{argument} entry {i} must be a (h_prev, h_curr) pair"
            )

        h_prev, h_curr = transition
        unknown = {h_prev, h_curr} - hidden_space

        if unknown:
            raise InvalidInputError(
                f"{argument} entry {i} contains labels that are not hidden "
                f"states of hidden_markov_model: {sorted(unknown, key=repr)}"
            )

        transition_collection.add((h_prev, h_curr))

    return transition_collection


# ------------------------------------------------------------------
# Base primitives
# ------------------------------------------------------------------


def mvr_constant(
    hidden_markov_model,
    value: bool,
) -> HomMVR:
    """Constructs the constantly true or constantly false MVR.

    This is the identity element for the boolean operators: ``mvr_and`` with a
    true constant and ``mvr_or`` with a false constant both leave the other
    operand's language unchanged. It is also the natural base case for building
    an MVR up by folding operators over a collection.

    The mediation space is a single absorbing state, so composing with a
    constant costs nothing.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
        Model whose hidden states define the MVR alphabet.
    value : bool
        Truth value the MVR evaluates to on every sequence.

    Returns
    -------
    HomMVR
        MVR accepting every nonempty sequence if ``value`` is ``True``, and no
        sequence otherwise.

    Raises
    ------
    InvalidInputError
        If the model has no hidden states or ``value`` is not a bool.
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
    """Constructs the MVR that accepts iff the last hidden state is in ``states``.

    This is the window-1 primitive, and the one most other constraints are built
    from. Because it reports a property of the current time step only, applying
    a temporal operator to it yields the corresponding global constraint::

        mvr_already_satisfied(mvr_current_state(hmm, states))   # ever visits
        mvr_not(mvr_already_satisfied(...))                     # never visits
        mvr_count(mvr_current_state(hmm, states), ">=2")        # visit count

    Evaluated at the end of a sequence it is exactly a "final state" constraint.

    The mediation space is two states, tracking whether the hidden state just
    consumed was in ``states``.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
        Model whose hidden states define the MVR alphabet.
    states : list, tuple, set, or frozenset
        Hidden-state labels the last hidden state must belong to. A single state
        must still be wrapped in a collection.

    Returns
    -------
    HomMVR
        MVR accepting the nonempty sequences whose last hidden state is in
        ``states``.

    Raises
    ------
    InvalidInputError
        If the model has no hidden states, or ``states`` is not a collection of
        hidden-state labels of the model.

    Examples
    --------
    >>> mvr = mvr_current_state(hmm, {"X"})  # doctest: +SKIP
    """
    hidden_states = _model_hidden_states(hidden_markov_model)
    state_collection = _validate_state_collection(
        states,
        hidden_states,
        argument="states",
    )

    # The mediation state is the truth value of the predicate on the hidden
    # state just consumed, so evl is the identity.
    mediation_states = [False, True]

    ini = {h: h in state_collection for h in hidden_states}

    upd = {
        (m, h): h in state_collection
        for m in mediation_states
        for h in hidden_states
    }

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
    """Constructs the MVR that accepts iff the last transition is in ``transitions``.

    This is the window-2 primitive. Adjacency cannot be recovered from
    ``mvr_current_state`` by any operator -- ``mvr_concatenate`` of two
    single-step MVRs recognizes "``h_curr`` sometime after ``h_prev``", not
    "``h_curr`` immediately after ``h_prev``" -- so this is a genuine primitive
    rather than a composition.

    A sequence of length one takes no transition and is therefore rejected.
    Combined with a temporal operator this gives the usual transition
    constraints::

        mvr_already_satisfied(mvr_transition(hmm, pairs))       # ever takes one
        mvr_not(mvr_already_satisfied(...))                     # forbidden pairs

    The mediation space is ``2 * len(hidden_states)`` states, tracking the hidden
    state just consumed together with the truth value of the predicate on the
    transition into it.

    Parameters
    ----------
    hidden_markov_model : HiddenMarkovModel
        Model whose hidden states define the MVR alphabet.
    transitions : list, tuple, set, or frozenset
        ``(h_prev, h_curr)`` pairs of hidden-state labels.

    Returns
    -------
    HomMVR
        MVR accepting the sequences of length at least two whose final adjacent
        pair is in ``transitions``.

    Raises
    ------
    InvalidInputError
        If the model has no hidden states, or ``transitions`` is not a
        collection of pairs of hidden-state labels of the model.

    Examples
    --------
    >>> mvr = mvr_transition(hmm, {("X", "Y")})  # doctest: +SKIP
    """
    hidden_states = _model_hidden_states(hidden_markov_model)
    transition_collection = _validate_transition_collection(
        transitions,
        hidden_states,
        argument="transitions",
    )

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
# mvr_operators.py. Nothing in this section builds ini/upd/evl by hand.
