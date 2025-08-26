from math import prod

from conin.util import try_import
from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor

with try_import() as pgmpy_available:
    import pgmpy.models


def _get_factor_value(f, values):
    """
    This is a wrapper for the DiscreteFactor.get_value() method.  This allows for
    the specification of node names that are non-strings.
    """
    return f.values[tuple(f.name_to_no[var][values[var]] for var in f.variables)]


class PgmpyWrapperDiscreteMarkovNetwork(DiscreteMarkovNetwork):

    def __init__(self, pgmpy_pgm):
        super().__init__()
        self._pgmpy_pgm = pgmpy_pgm
        self.states = pgmpy_pgm.states

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
        self.factors = factors


def convert_to_DiscreteMarkovNetwork(pgm):
    if (
        type(pgm) is DiscreteMarkovNetwork
        or type(pgm) is PgmpyWrapperDiscreteMarkovNetwork
    ):
        return pgm

    elif type(pgm) is pgmpy.models.MarkovNetwork:
        return PgmpyWrapperDiscreteMarkovNetwork(pgm)

    else:
        raise TypeError(f"Unexpected markov network type: {type(pgm)}")
