"""Total assignments and HMM sequences of the graphical models: their checks.

A sequence over the times ``0..T-1`` (HMM observations or hidden states) is a
non-empty list or tuple, or a dict keyed by exactly the times ``0..T-1``.
Anything else raises ``InvalidInputError``.  (The torch MVR algorithms also
accept a sparse ``{time: label}`` dict of observations; they check it
themselves.)

These helpers are internal: inference, scoring and the feasibility checks use
them so that every function reads an assignment or a sequence the same way.
"""

from conin.exceptions import InvalidInputError


def check_total_assignment(states, assignment, *, what="assignment"):
    """Check that ``assignment`` gives a valid state to exactly the nodes in ``states``.

    Parameters
    ----------
    states : dict
        Mapping from each node to its list of states.
    assignment : dict
        Mapping from nodes to states.
    what : str, optional
        Name used in error messages.

    Raises
    ------
    InvalidInputError
        If ``assignment`` is not a dict, misses a node, has an unknown node, or
        gives a node a state it does not have.
    """
    if not isinstance(assignment, dict):
        raise InvalidInputError(
            f"The {what} must be a dict, got {type(assignment).__name__}"
        )
    missing = [node for node in states if node not in assignment]
    if missing:
        raise InvalidInputError(
            f"The {what} must give a value to every node; missing: {missing}"
        )
    unknown = [node for node in assignment if node not in states]
    if unknown:
        raise InvalidInputError(f"The {what} has unknown nodes: {unknown}")
    for node, value in assignment.items():
        if value not in states[node]:
            raise InvalidInputError(
                f"In the {what}, {value!r} is not a state of node {node!r}; "
                f"expected one of {list(states[node])}"
            )


def dbn_horizon(dbn, assignment, *, what="assignment"):
    """Return ``stop`` for a total assignment of ``dbn`` over times ``0..stop``.

    The dynamic nodes are the ``(name, t)`` keys whose ``name`` is a dynamic
    node of ``dbn``.

    Raises
    ------
    InvalidInputError
        If ``assignment`` is not a dict, has no dynamic nodes, or its times are
        not ``0..stop``.
    """
    if not isinstance(assignment, dict):
        raise InvalidInputError(
            f"The {what} must be a dict, got {type(assignment).__name__}"
        )
    dynamic = set(dbn.dynamic_states)
    times = {
        key[1]
        for key in assignment
        if type(key) is tuple and len(key) == 2 and key[0] in dynamic
    }
    if not times:
        raise InvalidInputError(
            f"The {what} has no dynamic nodes (keys (name, t) with name in "
            f"{sorted(dynamic, key=str)})"
        )
    bad = [t for t in times if type(t) is not int]
    if bad:
        raise InvalidInputError(f"The {what} has non-integer times: {bad}")
    stop = max(times)
    if min(times) != 0 or len(times) != stop + 1:
        raise InvalidInputError(
            f"The {what} must cover the times 0..stop; got times {sorted(times)}"
        )
    return stop


def dbn_states(dbn, stop):
    """Map each node of ``dbn`` unrolled over ``0..stop`` to its states."""
    states = dict(dbn.states)
    for t in range(stop + 1):
        for name, s in dbn.dynamic_states.items():
            states[(name, t)] = s
    return states


def hmm_sequences(hmm, hidden, observed=None):
    """Return ``(hidden, observed)`` as lists of labels, checked against ``hmm``.

    ``hidden`` and ``observed`` are lists or tuples, or dicts keyed by
    ``0..T-1``.
    ``observed`` may be ``None``.

    Raises
    ------
    InvalidInputError
        For an empty or badly keyed sequence, unknown labels, or different
        lengths.
    """
    hidden = as_sequence(hidden, "hidden sequence")
    for h in hidden:
        if h not in hmm.hidden_to_internal:
            raise InvalidInputError(
                f"{h!r} is not a hidden state; expected one of {hmm.hidden_states}"
            )
    if observed is None:
        return hidden, None
    observed = as_sequence(observed, "observed sequence")
    if len(observed) != len(hidden):
        raise InvalidInputError(
            f"The hidden ({len(hidden)}) and observed ({len(observed)}) "
            "sequences must have the same length"
        )
    for o in observed:
        if o not in hmm.observed_to_internal:
            raise InvalidInputError(
                f"{o!r} is not an observed state; expected one of "
                f"{hmm.observed_states}"
            )
    return hidden, observed


def require_sequence_type(seq, what="observed sequence"):
    """Raise ``InvalidInputError`` unless ``seq`` is a list, a tuple or a dict.

    Only the type is checked (for the sparse ``{time: label}`` observations of
    the MVR algorithms); use :func:`as_sequence` for a dense sequence.
    """
    if isinstance(seq, (list, tuple, dict)):
        return
    raise InvalidInputError(_type_message(seq, what))


def as_sequence(seq, what="observed sequence"):
    """Return ``seq`` (a list, tuple or dict keyed by ``0..T-1``) as a list.

    Raises
    ------
    InvalidInputError
        For another type, a dict not keyed by exactly ``0..T-1``, or an empty
        sequence.
    """
    if isinstance(seq, dict):
        if not all(type(t) is int for t in seq) or set(seq) != set(range(len(seq))):
            raise InvalidInputError(
                f"The {what} must be keyed by the times 0..T-1 when given as a "
                f"dict; got the keys {list(seq)}"
            )
        seq = [seq[t] for t in range(len(seq))]
    elif isinstance(seq, (list, tuple)):
        seq = list(seq)
    else:
        raise InvalidInputError(_type_message(seq, what))
    if not seq:
        raise InvalidInputError(f"The {what} must not be empty")
    return seq


def _type_message(seq, what):
    return (
        f"The {what} must be a list, a tuple or a dict keyed by the times "
        f"0..T-1, got {type(seq).__name__}"
    )
