from conin.util import try_import
from conin.dynamic_bayesian_network.model import DynamicDiscreteBayesianNetwork
from conin.bayesian_network import DiscreteCPD

with try_import() as pgmpy_available:
    import pgmpy.models
    from pgmpy.models.DynamicBayesianNetwork import DynamicNode as pgmpy_DynamicNode


def _as_tuple(var, t, offset):
    if type(var):
        return (var[0], t + (var[1] - offset))
    return (var.node, t + (var.time_slice - offset))


class PgmpyWrapperDynamicDiscreteBayesianNetwork(DynamicDiscreteBayesianNetwork):

    def __init__(self, pgmpy_pgm):
        super().__init__()
        self._pgmpy_pgm = pgmpy_pgm

        tmp = {}
        for k, v in pgmpy_pgm.states.items():
            if type(k) is tuple:
                tmp[k[0]] = v
            elif type(k) is pgmpy_DynamicNode:
                tmp[k.node] = v
            else:
                tmp[k] = v
        self.dynamic_states = tmp

        cpds = []
        for cpd in pgmpy_pgm.get_cpds():

            if len(cpd.variables) == 1:  # evidence == []
                values = [v[0] for v in cpd.get_values()]

                cpds.append(
                    DiscreteCPD(
                        variable=_as_tuple(cpd.variable, 0, 0),
                        evidence=[],
                        values=values,
                    )
                )
            else:
                values = cpd.get_values()
                values = [
                    values[i][j]
                    for j in range(len(values[0]))
                    for i in range(len(values))
                ]

                if type(cpd.variable) is tuple:
                    offset = cpd.variable[1]
                else:
                    offset = cpd.variable.time_slice

                cpds.append(
                    DiscreteCPD(
                        variable=_as_tuple(cpd.variable, self.t, offset),
                        evidence=[
                            _as_tuple(var, self.t, offset) for var in cpd.variables[1:]
                        ],
                        values=values,
                    )
                )
        self.cpds = cpds


def convert_to_DynamicDiscreteBayesianNetwork(pgm):
    if (
        isinstance(pgm, DynamicDiscreteBayesianNetwork)
        or type(pgm) is PgmpyWrapperDynamicDiscreteBayesianNetwork
    ):
        return pgm

    elif type(pgm) is pgmpy.models.DynamicBayesianNetwork:
        return PgmpyWrapperDynamicDiscreteBayesianNetwork(pgm)

    else:
        raise TypeError(f"Unexpected Bayesian network type: {type(pgm)}")
