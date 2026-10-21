"""HMM observation (and hidden) sequences: the accepted forms and their checks.

A sequence over the times ``0..T-1`` is a non-empty list or tuple, or a dict
keyed by exactly the times ``0..T-1``.  Anything else raises
``InvalidInputError``.  (The torch MVR algorithms also accept a sparse
``{time: label}`` dict of observations; they check it themselves.)

These helpers are internal: inference, scoring and the feasibility checks use
them so that every query reads HMM observations the same way.
"""

from conin.exceptions import InvalidInputError


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


def time_range(observed, start, stop):
    """The times ``(start, stop)`` of a query on the observations ``observed`` (a list).

    ``stop`` defaults to the last observation, ``len(observed) - 1``.

    Raises
    ------
    InvalidInputError
        If ``stop`` is past the last observation.
    """
    last = len(observed) - 1
    if stop is None:
        return start, last
    if stop > last:
        raise InvalidInputError(
            f"stop={stop} is past the last observation (time {last})"
        )
    return start, stop


def unrolled_evidence(observed, start, stop):
    """The observations (a list) at the times ``start..stop`` as ``{("E", t): label}``.

    These are the evidence of the HMM unrolled as a DBN, whose observed nodes
    are ``("E", t)``.
    """
    return {("E", t): observed[t] for t in range(start, stop + 1)}


def _type_message(seq, what):
    return (
        f"The {what} must be a list, a tuple or a dict keyed by the times "
        f"0..T-1, got {type(seq).__name__}"
    )
