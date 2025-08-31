from math import prod

from conin.util import try_import
from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor
from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD
from conin.dynamic_bayesian_network.model import DynamicDiscreteBayesianNetwork

with try_import() as pgmpy_available:
    import pgmpy.models
    from pgmpy.models.DynamicBayesianNetwork import DynamicNode as pgmpy_DynamicNode


def _get_factor_value(f, values):
    """
    This is a wrapper for the DiscreteFactor.get_value() method.  This allows for
    the specification of node names that are non-strings.
    """
    return f.values[tuple(f.name_to_no[var][values[var]] for var in f.variables)]


def convert_pgmpy_to_DiscreteMarkovNetwork(pgmpy_pgm):
    # assert type(pgmpy_pgm) is pgmpy.models.MarkovNetwork, "Can only convert a pgmpy MarkovNetwork to a conin DiscreteMarkovNetwork"

    pgm = DiscreteMarkovNetwork()
    pgm._pgmpy_pgm = pgmpy_pgm
    pgm.states = pgmpy_pgm.states

    factors = []
    for factor in pgmpy_pgm.get_factors():
        vars = factor.scope()
        size = prod(factor.get_cardinality(vars).values())

        if len(vars) == 1:
            values = {
                key[0][1]: _get_factor_value(factor, dict(key))
                for key in factor.assignment(list(range(size)))
            }
        else:
            values = {
                tuple(v for _, v in key): _get_factor_value(factor, dict(key))
                for key in factor.assignment(list(range(size)))
            }

        factors.append(DiscreteFactor(nodes=factor.variables, values=values))
    pgm.factors = factors

    return pgm


def convert_pgmpy_to_DiscreteBayesianNetwork(pgmpy_pgm):
    # assert type(pgmpy_pgm) is pgmpy.models.DiscreteBayesianNetwork, "Can only convert a pgmpy DiscreteBayesianNetwork to a conin DiscreteBayesianNetwork"

    pgm = DiscreteBayesianNetwork()
    pgm._pgmpy_pgm = pgmpy_pgm
    pgm.states = pgmpy_pgm.states

    cpds = []
    for cpd in pgmpy_pgm.get_cpds():

        if len(cpd.variables) == 1:  # parents == []
            values = [v[0] for v in cpd.get_values()]

        else:
            values = cpd.get_values()
            values = [
                values[i][j] for j in range(len(values[0])) for i in range(len(values))
            ]

        cpds.append(
            DiscreteCPD(
                variable=cpd.variable,
                parents=[] if len(cpd.variables) == 1 else cpd.variables[1:],
                values=values,
            )
        )
    pgm.cpds = cpds

    return pgm


def _as_tuple(var, t, offset):
    if type(var):
        return (var[0], t + (var[1] - offset))
    return (var.node, t + (var.time_slice - offset))


def convert_pgmpy_to_DynamicDiscreteBayesianNetwork(pgmpy_pgm):

    pgm = DynamicDiscreteBayesianNetwork()
    pgm._pgmpy_pgm = pgmpy_pgm

    tmp = {}
    for k, v in pgmpy_pgm.states.items():
        if type(k) is tuple:
            tmp[k[0]] = v
        elif type(k) is pgmpy_DynamicNode:
            tmp[k.node] = v
        else:
            tmp[k] = v
    pgm.dynamic_states = tmp

    cpds = []
    for cpd in pgmpy_pgm.get_cpds():

        if len(cpd.variables) == 1:  # parents == []
            values = [v[0] for v in cpd.get_values()]

            cpds.append(
                DiscreteCPD(
                    variable=_as_tuple(cpd.variable, 0, 0),
                    parents=[],
                    values=values,
                )
            )
        else:
            values = cpd.get_values()
            values = [
                values[i][j] for j in range(len(values[0])) for i in range(len(values))
            ]

            if type(cpd.variable) is tuple:
                offset = cpd.variable[1]
            else:
                offset = cpd.variable.time_slice

            cpds.append(
                DiscreteCPD(
                    variable=_as_tuple(cpd.variable, pgm.t, offset),
                    parents=[
                        _as_tuple(var, pgm.t, offset) for var in cpd.variables[1:]
                    ],
                    values=values,
                )
            )
    pgm.cpds = cpds

    return pgm


def convert_pgmpy_to_conin(pgmpy_pgm):
    if type(pgmpy_pgm) is pgmpy.models.MarkovNetwork:
        return convert_pgmpy_to_DiscreteMarkovNetwork(pgmpy_pgm)

    elif type(pgmpy_pgm) is pgmpy.models.DiscreteBayesianNetwork:
        return convert_pgmpy_to_DiscreteBayesianNetwork(pgmpy_pgm)

    elif type(pgmpy_pgm) is pgmpy.models.DynamicBayesianNetwork:
        return convert_pgmpy_to_DynamicDiscreteBayesianNetwork(pgmpy_pgm)

    else:
        raise ValueError(f"Unexpected pgmpy model type: {type(pgmpy_pgm)}")
