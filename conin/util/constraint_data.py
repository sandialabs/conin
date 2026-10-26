"""The ``data`` object that constraint functions see."""

import copy

import munch


def dbn_time_fields(start, stop):
    """The fields of a DBN unrolled over ``start..stop``: ``Munch(start, stop, T)``.

    ``T`` is ``list(range(start, stop + 1))``.
    """
    return munch.Munch(start=start, stop=stop, T=list(range(start, stop + 1)))


def hmm_time_fields(start, stop, hidden_to_internal):
    """The fields of an HMM unrolled over ``start..stop``.

    ``Munch(start, stop, hmm=Munch(T, hidden_to_internal))``, with ``T`` as
    for :func:`dbn_time_fields` and a copy of ``hidden_to_internal``.
    """
    return munch.Munch(
        start=start,
        stop=stop,
        hmm=munch.Munch(
            T=list(range(start, stop + 1)),
            hidden_to_internal=dict(hidden_to_internal),
        ),
    )


def merge_constraint_data(app_data, fields):
    """Return the ``data`` passed to constraint functions for one query.

    Constraint functions get one ``data`` object: the application data of the
    constrained model (``Constrained*(model, constraints, data=...)``) with the
    fields a representation adds for the query (``T``, ``hmm``, ``evidence``,
    ...).

    Parameters
    ----------
    app_data : object or None
        The constrained model's ``data``.  It is never modified: the fields
        are set on a shallow copy (as items if it is a dict, e.g. a
        ``Munch``, otherwise as attributes).  A field replaces an application
        field of the same name.
    fields : munch.Munch or None
        The fields of the representation (``None`` if it adds none).

    Returns
    -------
    object
        ``fields`` itself if ``app_data`` is ``None``; otherwise the copy of
        ``app_data`` with the fields set.
    """
    if app_data is None:
        return fields
    data = copy.copy(app_data)
    for key, value in (fields or {}).items():
        if isinstance(data, dict):
            data[key] = value
        else:
            setattr(data, key, value)
    return data
