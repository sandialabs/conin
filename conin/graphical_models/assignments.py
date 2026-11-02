"""Total assignments of the graphical models: their checks and unrolled states.

These helpers are internal: scoring and the feasibility checks use them so
that every function reads an assignment the same way.
"""

from conin.exceptions import InvalidInputError
from conin.graphical_models.hmm_observations import as_sequence


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
