from dataclasses import dataclass
from math import log, prod
import numpy as np


@dataclass(order=True, frozen=True)
class State:
    value: tuple


def extract_factor_representation(pgm):
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
    S = {r:[State(value=s) for s in values] for r,values in pgm.states.items()}
    J = {}
    v = {}
    w = {}
    for factor in pgm.get_factors():
        vars = factor.scope()
        i = "_".join(vars)
        size = prod(factor.get_cardinality(vars).values())
        assignments = factor.assignment(list(range(size)))

        # J
        J[i] = list(range(size))

        # v
        for j, assignment in enumerate(assignments):
            if factor.get_value(**dict(assignment)) > 0:
                for key, value in assignment:
                    v[i, j, key] = State(value)

        # w
        values = [factor.get_value(**dict(assignment)) for assignment in assignments]
        total = np.sum(factor.values)
        # print("HERE",i,total,values)
        for j in range(size):
            if values[j] > 0:
                w[i, j] = log(values[j] / total)
            # j += 1     WEH - Why are we skipping every other value?
    return S, J, v, w

