"""Query checks shared by the integer-program and toulbar2 MAP queries.

Neither backend can run a partial MAP query (summing out the other variables
needs variable elimination), and neither supports pgmpy's network pruning or
Markov-network generation.
"""


def reject_pruning_options(options):
    """Pop ``prune_network`` and ``create_MN`` from ``options``; raise if either is set.

    Raises
    ------
    RuntimeError
        If ``prune_network`` or ``create_MN`` is true.
    """
    for option in ("prune_network", "create_MN"):
        if options.pop(option, False):
            raise RuntimeError(f"{option}=True is not supported for CONIN models")


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
