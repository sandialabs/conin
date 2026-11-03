"""The MAP query object, and the checks shared by the integer-program and toulbar2 solvers.

Neither backend can run a partial MAP query (summing out the other variables
needs variable elimination).
"""

import munch


def map_query_args(
    *,
    model=None,
    variables=None,
    evidence=None,
    observed=None,
    timing=False,
    start=0,
    stop=None,
    options=None,
):
    """The query that :func:`~conin.algorithms.inference.map_query` passes to the solvers.

    ``model`` is the queried model without its constraints (the solutions are
    scored on it).  ``evidence`` is the evidence as given; for an HMM,
    ``observed`` is the list of observations.  ``options`` are the method
    options (a new dict).
    """
    return munch.Munch(
        model=model,
        variables=variables,
        evidence=evidence,
        observed=observed,
        timing=timing,
        start=start,
        stop=stop,
        options=dict(options or {}),
    )


def full_query_variables(variables, nodes):
    """Return ``None`` if ``variables`` is empty or names every node of ``nodes``.

    Raises
    ------
    RuntimeError
        For a partial MAP query (any other ``variables``).
    """
    if not variables or set(variables) == set(nodes):
        return None
    raise RuntimeError(
        "Partial MAP queries are only supported by "
        "method='variable_elimination'; this method needs variables=None or "
        f"every node (got variables={list(variables)!r})"
    )
