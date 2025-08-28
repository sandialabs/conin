# import itertools
# import os
# from typing import Hashable, Optional, Dict, List, Tuple
# import warnings

from conin.util import try_import
from conin.bayesian_network.model import DiscreteBayesianNetwork, DiscreteCPD

with try_import() as pgmpy_available:
    import pgmpy.models

    # import pgmpy.factors.discrete


class PgmpyWrapperDiscreteBayesianNetwork(DiscreteBayesianNetwork):

    def __init__(self, pgmpy_pgm):
        super().__init__()
        self._pgmpy_pgm = pgmpy_pgm
        self.states = pgmpy_pgm.states

        cpds = []
        for cpd in pgmpy_pgm.get_cpds():

            if len(cpd.variables) == 1:  # evidence == []
                values = [v[0] for v in cpd.get_values()]

            else:
                values = cpd.get_values()
                values = [
                    values[i][j]
                    for j in range(len(values[0]))
                    for i in range(len(values))
                ]

            cpds.append(
                DiscreteCPD(
                    variable=cpd.variable,
                    evidence=[] if len(cpd.variables) == 1 else cpd.variables[1:],
                    values=values,
                )
            )
        self.cpds = cpds


def convert_to_DiscreteBayesianNetwork(pgm):
    if (
        isinstance(pgm, DiscreteBayesianNetwork)
        or type(pgm) is PgmpyWrapperDiscreteBayesianNetwork
    ):
        return pgm

    elif type(pgm) is pgmpy.models.DiscreteBayesianNetwork:
        return PgmpyWrapperDiscreteBayesianNetwork(pgm)

    else:
        raise TypeError(f"Unexpected Bayesian network type: {type(pgm)}")
