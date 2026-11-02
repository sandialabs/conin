"""Variable elimination on discrete factors, in log space.

The model is a list of :class:`~conin.graphical_models.DiscreteFactor` (with
dict ``values``, as in a :class:`~conin.graphical_models.DiscreteMarkovNetwork`)
and the states of its nodes.  Its weight is the product of the factors:

* :func:`max_product` returns the maximum weight over the query variables,
  with the other non-evidence variables summed out (a MAP query, or a
  partial MAP query), and an assignment that attains it;
* :func:`log_weight` returns the sum of the weights of the assignments that
  agree with the evidence (the partition function without evidence).

Each factor becomes a numpy array of log values (``-inf`` for ``0``), so
products are sums and nothing underflows.  The variables are eliminated in a
greedy order: next is the variable whose elimination creates the smallest
table.  A partial MAP query sums out the other variables before it maximises
over the query variables, so the tables that are maximised no longer depend
on the summed variables (the order is constrained, so a partial query can be
much more expensive than a full one).  Ties are broken deterministically
(``argmax`` takes the first state of the variable being eliminated), so
which of several optimal assignments is returned depends on the elimination
order.
"""

import heapq
import math

import numpy as np

from conin.exceptions import InvalidInputError


def max_product(factors, states, *, variables=None, evidence=None):
    """The maximum log weight over ``variables``, the others summed out.

    Parameters
    ----------
    factors : iterable of DiscreteFactor
        The factors of the model (dict ``values``; a missing entry has the
        factor's ``default_value``).
    states : dict
        ``{node: states}`` for every node of the factors.
    variables : list, optional
        The query variables (default: every node not in ``evidence``).  The
        nodes in neither ``variables`` nor ``evidence`` are summed out.
    evidence : dict, optional
        ``{node: state}``.

    Returns
    -------
    (float, dict)
        ``max_x log sum_y prod f(x, y, e)`` over the assignments ``x`` of
        ``variables`` (``y`` are the other non-evidence nodes), and an
        assignment ``{variable: state}`` that attains it.  If the value is
        ``-inf`` (every assignment has weight zero), the assignment is
        arbitrary.

    Raises
    ------
    InvalidInputError
        For an unknown node or state, or a variable that is also evidence.
    """
    evidence = dict(evidence or {})
    index = _evidence_index(states, evidence)
    if variables is None:
        variables = [node for node in states if node not in evidence]
    else:
        variables = list(variables)
        _check_variables(states, variables, evidence)
    query = set(variables)
    summed = [node for node in states if node not in evidence and node not in query]

    tables = _tables(factors, states, index, [*summed, *variables])
    tables, _ = _eliminate(tables, summed, maximise=False)
    tables, argmaxes = _eliminate(tables, variables, maximise=True)
    value = _total(tables)

    # Trace back: each variable's argmax depends on the variables eliminated
    # after it.
    chosen = {}
    for node, others, argmax in reversed(argmaxes):
        chosen[node] = int(argmax[tuple(chosen[other] for other in others)])
    return value, {node: states[node][chosen[node]] for node in variables}


def log_weight(factors, states, *, evidence=None):
    """``log`` of the sum of the weights of the assignments that agree with ``evidence``.

    Parameters are as for :func:`max_product`.  Without evidence this is the
    log partition function (``0`` for a Bayesian network).  Returns ``-inf``
    if every such assignment has weight zero.
    """
    evidence = dict(evidence or {})
    index = _evidence_index(states, evidence)
    summed = [node for node in states if node not in evidence]
    tables = _tables(factors, states, index, summed)
    tables, _ = _eliminate(tables, summed, maximise=False)
    return _total(tables)


# ---------------------------------------------------------------------
# Tables: (nodes, array of log values with one axis per node)
# ---------------------------------------------------------------------


def _evidence_index(states, evidence):
    """``{node: index of its state}`` for the evidence."""
    index = {}
    for node, state in evidence.items():
        if node not in states:
            raise InvalidInputError(f"Unknown evidence variable {node!r}")
        try:
            index[node] = list(states[node]).index(state)
        except ValueError:
            raise InvalidInputError(
                f"Unknown state {state!r} of the evidence variable {node!r} "
                f"(its states are {list(states[node])})"
            ) from None
    return index


def _check_variables(states, variables, evidence):
    for node in variables:
        if node not in states:
            raise InvalidInputError(f"Unknown query variable {node!r}")
        if node in evidence:
            raise InvalidInputError(
                f"The variable {node!r} is both a query variable and evidence"
            )


def _tables(factors, states, index, eliminated):
    """The log tables of ``factors`` with the evidence ``index`` fixed.

    A node in ``eliminated`` that no factor covers gets a table of zeros (a
    factor ``1``), so it is summed (or maximised) over its states.
    """
    tables = [_reduce(_log_table(factor, states), index) for factor in factors]
    covered = {node for nodes, _ in tables for node in nodes}
    for node in eliminated:
        if node not in covered:
            tables.append(((node,), np.zeros(len(states[node]))))
    return tables


def _log_table(factor, states):
    nodes = tuple(factor.nodes)
    positions = [{state: i for i, state in enumerate(states[node])} for node in nodes]
    values = np.full(
        tuple(len(states[node]) for node in nodes), float(factor.default_value)
    )
    for key, value in factor.values.items():
        if len(nodes) == 1:
            key = (key,)
        values[tuple(pos[state] for pos, state in zip(positions, key))] = value
    with np.errstate(divide="ignore"):
        return nodes, np.log(values)


def _reduce(table, index):
    """Fix the evidence nodes of ``table`` to their states."""
    nodes, values = table
    if not any(node in index for node in nodes):
        return table
    key = tuple(index.get(node, slice(None)) for node in nodes)
    return tuple(node for node in nodes if node not in index), values[key]


def _product(tables):
    """The product (sum of log values) of ``tables``, over the union of their nodes."""
    nodes = []
    for table_nodes, _ in tables:
        nodes.extend(node for node in table_nodes if node not in nodes)
    position = {node: i for i, node in enumerate(nodes)}
    total = 0.0
    for table_nodes, values in tables:
        order = sorted(range(len(table_nodes)), key=lambda i: position[table_nodes[i]])
        shape = [1] * len(nodes)
        for i in order:
            shape[position[table_nodes[i]]] = values.shape[i]
        total = total + np.transpose(values, order).reshape(shape)
    return tuple(nodes), total


def _logsumexp(values, axis):
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(np.exp(values - peak), axis=axis))
    return total + np.squeeze(peak, axis=axis)


def _eliminate(tables, nodes, *, maximise):
    """Sum out (or maximise over) ``nodes``, in a greedy order.

    The next node is the one whose elimination creates the smallest table
    (ties: the first in ``nodes``).  Eliminating a node only changes the
    sizes of its neighbours, so a heap keeps the sizes, and stale entries are
    skipped.

    Returns the remaining tables and, if ``maximise``, the list of
    ``(node, other nodes, argmax array over the other nodes)`` in
    elimination order.
    """
    tables = dict(enumerate(tables))
    touching = {}
    for key, (table_nodes, _) in tables.items():
        for node in table_nodes:
            touching.setdefault(node, set()).add(key)
    rank = {node: i for i, node in enumerate(nodes)}
    size = {node: _elimination_size(tables, touching, node) for node in nodes}
    heap = [(size[node], rank[node], node) for node in nodes]
    heapq.heapify(heap)
    next_key = len(tables)
    argmaxes = []
    while heap:
        node_size, _, node = heapq.heappop(heap)
        if node not in size or size[node] != node_size:
            continue  # eliminated, or a stale size
        del size[node]
        removed = touching.pop(node)
        product_nodes, values = _product([tables.pop(key) for key in sorted(removed)])
        for other in product_nodes:
            if other != node:
                touching[other] -= removed
        axis = product_nodes.index(node)
        others = product_nodes[:axis] + product_nodes[axis + 1 :]
        if maximise:
            argmaxes.append((node, others, np.argmax(values, axis=axis)))
            values = np.max(values, axis=axis)
        else:
            values = _logsumexp(values, axis=axis)
        tables[next_key] = (others, values)
        for other in others:
            touching[other].add(next_key)
            if other in size:
                size[other] = _elimination_size(tables, touching, other)
                heapq.heappush(heap, (size[other], rank[other], other))
        next_key += 1
    return list(tables.values()), argmaxes


def _elimination_size(tables, touching, node):
    """The size of the table that eliminating ``node`` creates."""
    sizes = {}
    for key in touching.get(node, ()):
        table_nodes, values = tables[key]
        sizes.update(zip(table_nodes, values.shape))
    sizes.pop(node, None)
    return math.prod(sizes.values())


def _total(tables):
    """The sum of the remaining tables, which have no nodes left."""
    return float(sum(float(values) for _, values in tables))
