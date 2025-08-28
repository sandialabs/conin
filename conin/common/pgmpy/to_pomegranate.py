import numpy as np
import itertools
from munch import Munch

from conin.util import try_import


def convert_pgmpy_to_pomegranate(pgm):
    """
    Converts a pgmpy model to a pomegranate model

    pomegrante models lose a lot of information compared to pgmpy models.
    We store the following in output.extra_data (which is a munch object):
        - distributions: list[pomegranate.distribution]
        - name_to_distribution: dict[str, pomegranate.distribution]
        - distribution_to_cat_values: dict[pomegranate.distribution, list[str]]
        - distribution_to_parent_names: dict[pomegranate.distribution, list[str]]
        - distribution_to_name: dict[pomegranate.distribution, str]

    Inputs:
        - pgm: pgmpy probabalistic model

    Returns:
        - pomegranate model
    """
    with try_import() as pomegranate_available:
        from pomegranate.distributions import Categorical, ConditionalCategorical
        from pomegranate.bayesian_network import BayesianNetwork

    if not pomegranate_available:
        raise ImportError("Cannot convert to a pomegranatepgmax model without importing pomegranate")

    distributions = []
    name_to_distribution = {}
    distribution_to_cat_values = {}
    distribution_to_parent_names = {}
    distribution_to_name = {}
    distribution_to_cards = {}
    _name_and_cat_value_to_index = {}
    _name_to_cat_values = {}

    for cpd in pgm.get_cpds():
        name = cpd.variable
        cat_values = cpd.state_names[name]
        _name_to_cat_values[name] = frozenset(cat_values)

        for i, cat in enumerate(cat_values):
            _name_and_cat_value_to_index[(name, cat)] = i

    for cpd in pgm.get_cpds():
        name = cpd.variable
        cat_values = cpd.state_names[name]
        parents = cpd.variables[1:]  # The first one is the actual variable

        if len(cpd.variables) == 1:
            dis = Categorical([cpd.values])
            cards = [len(cat_values)]
        else:
            df = cpd.to_dataframe()
            all_cat_values = [list(_name_to_cat_values[name]) for name in parents]
            cards = [len(vec) for vec in all_cat_values]
            cards.append(len(cat_values))
            table = np.zeros(cards)

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
                    table_indices.append(i)
                    table[tuple(table_indices)] = df.loc[df_tuple]

            dis = ConditionalCategorical([table])

        distributions.append(dis)

        name_to_distribution[name] = dis
        distribution_to_cat_values[dis] = cat_values
        distribution_to_parent_names[dis] = parents
        distribution_to_name[dis] = name
        distribution_to_cards[dis] = tuple(cards)

    output = BayesianNetwork(distributions)

    # Here, since everything is not named, the order that you add edges matters...
    for dis in distributions:
        for dis_parent_name in distribution_to_parent_names[dis]:
            output.add_edge(name_to_distribution[dis_parent_name], dis)

    extra_data = Munch()
    extra_data.names_to_distribution = name_to_distribution
    extra_data.distribution_to_cat_values = distribution_to_cat_values
    extra_data.distribution_to_parent_names = distribution_to_parent_names
    extra_data.distributions = distributions
    extra_data.distribution_to_name = distribution_to_name
    extra_data.disribution_to_cards = distribution_to_cards
    output.extra_data = extra_data

    return output

