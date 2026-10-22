"""The factor representation ``(S, J, v, w)`` of a Markov network.

Also the helpers that build it for the IP formulations: the BN -> MN skeleton
(:func:`skeleton_mn_of_bn`), the tuple-node name map
(:func:`tuple_var_index_map`) and :func:`factor_representation_of_mn`.
"""

from math import log
from conin.exceptions import InvalidInputError
from conin.util import State
from conin.graphical_models import DiscreteMarkovNetwork


def _check_unique_variable_names(pgm_states, var_index_map):
    """Raise if two nodes are mapped to the same pyomo variable name."""
    # TODO: Use clash-free (e.g. integer) variable ids instead of raising.
    seen = {}
    for r in pgm_states:
        name = var_index_map[r]
        if name in seen:
            raise InvalidInputError(
                f"Nodes {seen[name]!r} and {r!r} both have the internal "
                f"variable name {name!r}; rename one of them to use the "
                "pyomo backend"
            )
        seen[name] = r


def get_factor_value(f, values):
    """
    This is a wrapper for the DiscreteFactor.get_value() method.  This allows for
    the specification of node names that are non-strings.
    """
    return f.values[tuple(f.name_to_no[var][values[var]] for var in f.variables)]


def extract_factor_representation(pgm):
    return extract_factor_representation_(pgm.states, pgm.factors)


def extract_factor_representation_(pgm_states, pgm_factors, var_index_map=None):
    #
    # S[r]: the (finite) set of possible values of variable X_r
    #           Variable values s can be integers or strings
    #
    # J[i]: the (finite) set of possible configurations (rows) of factor i
    #           J[i] contains the configuration ids for factor i
    #
    # v[i,j,r]: the value of variable r in row j of factor i
    #           Note that v[i,j,r] \in S[r]
    #           Note that j \in J[i]
    #
    # w[i,j]: the log-probability of factor i in configuration j
    #           Note that j \in J[i]
    #
    if var_index_map is None:
        var_index_map = {}

    if var_index_map:
        _check_unique_variable_names(pgm_states, var_index_map)
        S = {
            var_index_map[r]: [State(value=s) for s in values]
            for r, values in pgm_states.items()
        }
    else:
        S = {r: [State(value=s) for s in values] for r, values in pgm_states.items()}
    J = {}
    v = {}
    w = {}
    factor_nodes = {}
    for factor in pgm_factors:
        # Create a string name for this factor
        vars = factor.nodes
        if var_index_map:
            i = "_".join(var_index_map[v] for v in vars)
        else:
            i = "_".join(vars)
        # Factors over different nodes must not share a name.  (Factors over
        # the same nodes still overwrite each other, as before.)
        if i in factor_nodes and factor_nodes[i] != list(vars):
            # TODO: Use clash-free (e.g. integer) factor ids instead of raising.
            raise InvalidInputError(
                f"Factors over {factor_nodes[i]} and {list(vars)} both have "
                f"the internal name {i!r}; the pyomo backend does not support "
                "factors with the same name"
            )
        factor_nodes[i] = list(vars)

        # J
        size = len(factor.values)
        J[i] = list(range(size))

        if type(factor.values) is dict:
            indices = list(sorted(factor.values.keys()))

        # Compute values of assignments, normalizing if all values are zero
        if type(factor.values) is dict:
            values = [factor.values[index] for index in indices]
        else:
            values = factor.values
        total = sum(values)
        if total == 0.0:
            values = [1 / len(values)] * size
            total = 1.0

        # v
        if type(factor.values) is dict:
            if len(vars) == 1:
                for j, assignment in enumerate(indices):
                    value = factor.values[assignment]
                    if values[j] > 0:
                        for key in vars:
                            value = assignment
                            key = var_index_map.get(key, key)
                            v[i, j, key] = State(value)
            else:
                for j, assignment in enumerate(indices):
                    value = factor.values[assignment]
                    if values[j] > 0:
                        for k, key in enumerate(vars):
                            value = assignment[k]
                            key = var_index_map.get(key, key)
                            v[i, j, key] = State(value)
        else:
            for j, assignment in enumerate(factor.assignments(pgm_states)):
                if values[j] > 0:
                    for key, value in assignment:
                        key = var_index_map.get(key, key)
                        v[i, j, key] = State(value)

        # w
        for j in range(size):
            if values[j] > 0:
                w[i, j] = log(values[j] / total)

    return S, J, v, w


def factor_representation_of_mn(mn):
    """Return ``(S, J, v, w, var_index_map)`` for a :class:`DiscreteMarkovNetwork`.

    Tuple nodes (e.g. the time-indexed nodes of an unrolled DBN) are mapped to
    flat names like ``"A_0"``; other nodes are used as-is.
    """
    var_index_map = tuple_var_index_map(mn.states)
    S, J, v, w = extract_factor_representation_(mn.states, mn.factors, var_index_map)
    return S, J, v, w, var_index_map


def skeleton_mn_of_bn(bn):
    """The Markov network skeleton used for the IP of a Bayesian network.

    It has one factor per CPD (no moralisation or variable elimination).
    """
    return DiscreteMarkovNetwork(
        states=bn.states, factors=[cpd.to_factor() for cpd in bn.cpds]
    )


def tuple_var_index_map(states):
    """Map tuple nodes (e.g. ``("A", 0)``) to flat names (``"A_0"``).

    Returns ``None`` when no node is a tuple, so the nodes are used as-is.
    """
    nodes_ = list(states.keys())
    if any(type(v) is tuple for v in nodes_):
        return {
            v: "_".join(str(i) for i in v) if type(v) is tuple else v for v in nodes_
        }
    return None
