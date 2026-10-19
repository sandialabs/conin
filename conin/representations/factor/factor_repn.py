from math import log
from conin.exceptions import InvalidInputError
from conin.util import State


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
