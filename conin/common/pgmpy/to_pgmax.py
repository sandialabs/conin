import numpy as np
import itertools

from conin.util import try_import


def Log(x):
    if x == 0.0:
        return -np.inf
    return np.log(x)


def convert_pgmpy_to_pgmax(pgm):
    """
    Takes in a pgmpy model and return a pgmax model.
    pgmax only works with factor graphs so this is a different
    type of conversion than pgmpy_to_pomegranate.

    Inputs:
        - pgm: pgmpy probabalistic model

    Returns:
        - pgmax model
    """
    with try_import() as pgmax_available:
        from pgmax import fgraph, factor, vgroup

    if not pgmax_available:
        raise ImportError("Cannot convert to a pgmax model without importing pgmax")

    _name_and_cat_value_to_index = {}
    name_to_cat_values = {}
    name_and_index_to_cat_value = {}
    name_and_cat_value_to_index = {}

    for cpd in pgm.get_cpds():
        name = cpd.variable
        cat_values = cpd.state_names[name]
        name_to_cat_values[name] = frozenset(cat_values)

        for i, cat in enumerate(cat_values):
            _name_and_cat_value_to_index[(name, cat)] = i
            name_and_index_to_cat_value[(name, i)] = cat
            name_and_cat_value_to_index[(name, cat)] = i

    num_states = []
    names = []
    parents_list = []
    log_likelihoods = []

    # TODO CLM: They use log potentials in pgmax
    # I am just assuming that we can use log likelihoods,
    # but we should double check that.
    for cpd in pgm.get_cpds():
        name = cpd.variable
        names.append(name)
        num_states.append(len(cpd.state_names[name]))
        parents = cpd.variables[1:]
        parents_list.append(parents)
        cat_values = cpd.state_names[name]

        if len(cpd.variables) == 1:
            inner_log_likelihood = np.array([Log(val) for val in cpd.values])
        else:
            df = cpd.to_dataframe()
            inner_log_likelihood = np.zeros(cpd.cardinality)
            all_cat_values = [list(name_to_cat_values[parent]) for parent in parents]

            for index in itertools.product(*all_cat_values):
                for i, cat_value in enumerate(cat_values):
                    df_tuple = []
                    table_indices = []
                    for j, val in enumerate(index):
                        table_indices.append(
                            _name_and_cat_value_to_index[(parents[j], val)]
                        )
                        df_tuple.append(val)
                    df_tuple = (tuple(df_tuple), cat_value)
                    table_indices.insert(
                        0, i
                    )  # The original vertex is at the beginning (note that this is opposite of df_tuple...
                    inner_log_likelihood[tuple(table_indices)] = Log(df.loc[df_tuple])

        log_likelihoods.append(inner_log_likelihood)

    pgmax_variables = vgroup.VarDict(
        num_states=np.array(num_states), variable_names=names
    )
    fg = fgraph.FactorGraph(variable_groups=pgmax_variables)

    # Similar to pomegranate the order you do this matters

    for i, name in enumerate(names):
        variables_for_factor = [pgmax_variables[parent] for parent in parents_list[i]]
        variables_for_factor.insert(0, pgmax_variables[name])

        factor_configs = []
        log_potentials = []

        for index, val in np.ndenumerate(np.array(log_likelihoods[i])):
            if not isinstance(index, int):
                index = list(index)
            factor_configs.append(index)
            log_potentials.append(val)

        fac = factor.EnumFactor(
            variables=variables_for_factor,
            log_potentials=np.array(log_potentials, dtype=float),
            factor_configs=np.array(factor_configs, dtype=int),
        )

        fg.add_factors(fac)

    fg.name_and_index_to_cat_values = name_and_index_to_cat_value
    fg.name_and_cat_value_to_index = name_and_cat_value_to_index
    fg.name_to_cat_values = name_to_cat_values
    return fg
