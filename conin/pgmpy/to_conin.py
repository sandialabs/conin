import math
from math import prod

from conin.exceptions import InvalidInputError
from conin.util import try_import
from conin.graphical_models import DiscreteMarkovNetwork, DiscreteFactor
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
from conin.graphical_models import DynamicDiscreteBayesianNetwork
from conin.graphical_models import ExpressionVariable

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

    pgm = DiscreteMarkovNetwork(states=pgmpy_pgm.states, factors=factors)

    return pgm


def convert_pgmpy_to_DiscreteBayesianNetwork(pgmpy_pgm):
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
                node=cpd.variable,
                parents=[] if len(cpd.variables) == 1 else cpd.variables[1:],
                values=values,
            )
        )

    pgm = DiscreteBayesianNetwork(states=pgmpy_pgm.states, cpds=cpds)

    return pgm


def _same_values(a, b):
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15) for x, y in zip(a, b)
    )


def _as_tuple(var, t, offset):
    # A pgmpy DBN node is a (name, time_slice) tuple or a DynamicNode; both
    # support indexing.
    return (var[0], t + (var[1] - offset))


def convert_pgmpy_to_DynamicDiscreteBayesianNetwork(pgmpy_pgm):

    tmp = {}
    for k, v in pgmpy_pgm.states.items():
        if type(k) is tuple:
            tmp[k[0]] = v
        elif type(k) is pgmpy_DynamicNode:
            tmp[k.node] = v
        else:
            tmp[k] = v

    t = ExpressionVariable()

    cpds = []
    templates = {}
    for cpd in pgmpy_pgm.get_cpds():

        if len(cpd.variables) == 1:  # parents == []
            values = [v[0] for v in cpd.get_values()]

            cpds.append(
                DiscreteCPD(
                    node=_as_tuple(cpd.variable, 0, 0),
                    parents=[],
                    values=values,
                )
            )
        else:
            values = cpd.get_values()
            values = [
                values[i][j] for j in range(len(values[0])) for i in range(len(values))
            ]

            offset = cpd.variable[1]

            # pgmpy may hold the same CPD for several time slices (e.g.
            # initialize_initial_state adds the slice-1 copy of an intra-slice
            # CPD). They give the same (name, t) template, so keep one: two
            # templates for one node would give the unrolled network two CPDs
            # per node.
            key = tuple((var[0], var[1] - offset) for var in cpd.variables)
            if key in templates:
                if not _same_values(templates[key], values):
                    raise InvalidInputError(
                        f"The pgmpy CPDs of {cpd.variable!r} and of another time "
                        f"slice give the same dynamic CPD template {key!r} with "
                        f"different values"
                    )
                continue
            templates[key] = values

            cpds.append(
                DiscreteCPD(
                    node=_as_tuple(cpd.variable, t, offset),
                    parents=[_as_tuple(var, t, offset) for var in cpd.variables[1:]],
                    values=values,
                )
            )

    pgm = DynamicDiscreteBayesianNetwork(dynamic_states=tmp, cpds=cpds, t=t)

    return pgm


def convert_pgmpy_to_conin(pgmpy_pgm):
    """Convert a pgmpy model to the matching CONIN model.

    Parameters
    ----------
    pgmpy_pgm : pgmpy.models.DiscreteMarkovNetwork or DiscreteBayesianNetwork or DynamicBayesianNetwork
        The model to convert.

    Returns
    -------
    DiscreteMarkovNetwork or DiscreteBayesianNetwork or DynamicDiscreteBayesianNetwork
        The CONIN model. A dynamic Bayesian network's nodes are ``(name, t)``;
        CPDs that pgmpy holds for several time slices become one CPD template.

    Raises
    ------
    ImportError
        If pgmpy is not installed.
    ValueError
        For another model type.
    InvalidInputError
        If two time slices of a dynamic Bayesian network have different CPDs
        for the same template.
    """
    if not pgmpy_available:
        raise ImportError("Cannot convert a pgmpy model without importing pgmpy")

    if type(pgmpy_pgm) is pgmpy.models.DiscreteMarkovNetwork:
        return convert_pgmpy_to_DiscreteMarkovNetwork(pgmpy_pgm)

    elif type(pgmpy_pgm) is pgmpy.models.DiscreteBayesianNetwork:
        return convert_pgmpy_to_DiscreteBayesianNetwork(pgmpy_pgm)

    elif type(pgmpy_pgm) is pgmpy.models.DynamicBayesianNetwork:
        return convert_pgmpy_to_DynamicDiscreteBayesianNetwork(pgmpy_pgm)

    else:
        raise ValueError(f"Unexpected pgmpy model type: {type(pgmpy_pgm)}")
