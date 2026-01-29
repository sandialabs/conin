from conin.bayesian_network.model import DiscreteBayesianNetwork
from conin.dynamic_bayesian_network.model import DynamicDiscreteBayesianNetwork
from conin.markov_network.model import DiscreteMarkovNetwork


def save_model(pgm, name, quiet=True):
    if name.endswith(".uai"):
        return save_model_uai(pgm, name, quiet)

    raise RuntimeError(
        f"Cannot save conin model.  Uknown format specified by filename suffix: {name}"
    )


def save_model_uai(pgm, name, quiet=True):
    """
    Function to serialize the parameters of a DBN in UAI format
    Inputs:  pgm (<Dynamic>DiscreteBayesianNetwork, DiscreteMarkovNetwork) - model to be converted
             name  (str) - filename of output (will end in .uai)
             quiet (bool) - unused
    Outputs: <name>.uai - written UAI file
    """
    if isinstance(pgm, DiscreteBayesianNetwork):
        case = "BAYES"
    elif isinstance(pgm, DynamicDiscreteBayesianNetwork):
        case = "BAYES"
    elif isinstance(pgm, DiscreteMarkovNetwork):
        case = "MARKOV"
    else:
        raise ValueError(
            f"pgm must be an instance of a <Dynamic>DiscreteBayesianNetwork or DiscreteMarkovNetwork: {type(pgm)=}"
        )

    # get node2id mapping
    node2id = {n: i for i, n in enumerate(pgm.nodes)}

    # get domain sizes
    sizes = [str(len(pgm.states[n])) for n in pgm.nodes]

    # get factors
    if case == "BAYES":
        factors = [cpd.to_factor() for cpd in pgm.cpds]
    elif case == "MARKOV":
        factors = pgm.factors

    # get function table preamble and values for each vactor
    tables = []
    for factor in factors:
        # table_preamble
        # len(factor.nodes), factor.node[0], factor.node[1], ..., factor.node[n-1]
        factor_node_ids = [str(node2id[n]) for n in factor.nodes]
        table_preamble = [str(len(factor_node_ids))] + factor_node_ids

        # table values
        assignments = list(factor.assignments(pgm.states))
        table_values = []
        for assignment in assignments:
            state = tuple([t[1] for t in assignment])
            table_values.append(str(factor.values[state]))

        tables += [(table_preamble, table_values)]

    # open file
    with open(name.split(".")[0] + ".uai", "w") as f:
        # Preamble
        f.write(case + "\n")  # network type
        f.write(str(len(node2id)) + "\n")  # num. nodes
        f.write(" ".join(sizes) + "\n")  # node domain sizes
        f.write(str(len(tables)) + "\n")  # num. function tables
        for table_preamble, _ in tables:
            f.write(" ".join(table_preamble) + "\n")  # table preamble

        # Function Tables
        f.write("\n")
        for _, table_values in tables:
            f.write(str(len(table_values)) + "\n")  # len table
            f.write(" ".join(table_values) + "\n")  # table values
