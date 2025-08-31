import copy
from conin.bayesian_network import DiscreteCPD, DiscreteBayesianNetwork


def all_cpds(*args):
    for arg in args:
        if arg:
            for v in arg:
                yield v


def create_bn_from_dbn(*, dbn, start, stop):
    assert start < stop

    #
    # Copy non-dynamic states
    #
    states = copy.copy(dbn.states)
    _pyomo_index_names = {key: key for key in states}

    #
    # Copy dynamic states to all time steps
    #
    for t in range(start, stop + 1):
        for v, s in dbn.dynamic_states.items():
            states[(v, t)] = s
            _pyomo_index_names[(v, t)] = f"{v}_{t}"

    #
    # Copy cpds
    #
    # If the variable or parents are dynamic tuples, then we treat the
    # cpd as a dynamic map and add it if the time steps for all dynamic variaables
    # are feasible.
    #
    cpds = []
    for cpd in dbn.cpds:
        dynamic = False
        for v in all_cpds([cpd.variable], cpd.parents):
            if (
                type(v) is tuple
                and v[0] in dbn.dynamic_states
                and not (type(v[1]) is int)
            ):
                dynamic = True
                break

        if not dynamic:
            cpds.append(cpd)
            continue

        for t in range(start, stop + 1):
            dbn.t.set_value(t)

            if type(cpd.variable) is tuple:
                curr = cpd.variable[1].value()
                if curr < start:
                    continue
                variable = (cpd.variable[0], curr)
            else:
                variable = cpd.variable

            skip = False
            parents = []
            for v in all_cpds(cpd.parents):
                if type(v) is tuple:
                    curr = v[1].value()
                    if curr < start:
                        skip = True
                        break
                    parents.append((v[0], curr))
                else:
                    parents.append(v)

            if skip:
                continue
            cpds.append(
                DiscreteCPD(variable=variable, parents=parents, values=cpd.values)
            )

    pgm = DiscreteBayesianNetwork()
    pgm.states = states
    pgm.cpds = cpds
    pgm.check_model()
    pgm._pyomo_index_names = _pyomo_index_names

    return pgm
