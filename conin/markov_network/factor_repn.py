from dataclasses import dataclass
from math import log, prod
import numpy as np


def get_factor_value(f, values):
    """
    This is a wrapper for the DiscreteFactor.get_value() method.  This allows for
    the specification of node names that are non-strings.
    """
    return f.values[tuple(f.name_to_no[var][values[var]] for var in f.variables)]


@dataclass(order=True, frozen=True)
class State:
    value: tuple


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
        S = {
            var_index_map[r]: [State(value=s) for s in values]
            for r, values in pgm_states.items()
        }
    else:
        S = {r: [State(value=s) for s in values] for r, values in pgm_states.items()}
    J = {}
    v = {}
    w = {}
    for factor in pgm_factors:
        # Create a string name for this factor
        vars = factor.nodes
        if var_index_map:
            i = "_".join(var_index_map[v] for v in vars)
        else:
            i = "_".join(vars)

        # J
        size = len(factor.values)
        J[i] = list(range(size))

        # Compute values of assignments, normalizing if all values are zero
        if type(factor.values) is dict:
            values = [v for _, v in factor.values.items()]
        else:
            values = factor.values
        total = sum(values)
        if total == 0.0:
            values = [1 / len(values)] * size
            total = 1.0

        # v
        if type(factor.values) is dict:
            if len(vars) == 1:
                for j, (assignment, value) in enumerate(factor.values.items()):
                    if values[j] > 0:
                        for key in vars:
                            value = assignment
                            key = var_index_map.get(key, key)
                            v[i, j, key] = State(value)
            else:
                for j, (assignment, value) in enumerate(factor.values.items()):
                    if values[j] > 0:
                        for k, key in enumerate(vars):
                            value = assignment[k]
                            key = var_index_map.get(key, key)
                            v[i, j, key] = State(value)
        else:
            for j, assignment in enumerate(factor.assignments()):
                if values[j] > 0:
                    for key, value in assignment:
                        key = var_index_map.get(key, key)
                        v[i, j, key] = State(value)

        # w
        for j in range(size):
            if values[j] > 0:
                w[i, j] = log(values[j] / total)

    return S, J, v, w
