import itertools
from conin.util import try_import
from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD

with try_import() as pgmpy_available:
    from pgmpy.models import DiscreteBayesianNetwork as pgmpy_DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD as pgmpy_TabularCPD


def convert_conin_to_pgmpy(conin_pgm, check_model=True):
    """
    Convert a conin DiscreteBayesianNetwork to a pgmpy DiscreteBayesianNetwork.

    Parameters
    ----------
    conin_pgm : DiscreteBayesianNetwork
        The conin Bayesian network to convert

    Returns
    -------
    pgmpy_DiscreteBayesianNetwork
        The converted pgmpy Bayesian network

    Raises
    ------
    ImportError
        If pgmpy is not available
    ValueError
        If the input is not a conin DiscreteBayesianNetwork
    """
    if not pgmpy_available:
        raise ImportError("Cannot convert to pgmpy without pgmpy installed")

    if not isinstance(conin_pgm, DiscreteBayesianNetwork):
        raise ValueError(
            f"Expected conin DiscreteBayesianNetwork, got {type(conin_pgm)}"
        )

    # Create pgmpy model
    pgmpy_pgm = pgmpy_DiscreteBayesianNetwork()

    # Add nodes
    for node in conin_pgm.nodes:
        pgmpy_pgm.add_node(node)

    # Convert CPDs and build edges
    pgmpy_cpds = []
    edges = set()

    for conin_cpd in conin_pgm.cpds:
        # Convert CPD
        pgmpy_cpd = _convert_conin_cpd_to_pgmpy_cpd(conin_cpd, conin_pgm)

        # Track edges for parent-child relationships
        if conin_cpd.parents:
            for parent in conin_cpd.parents:
                edges.add((parent, conin_cpd.node))

        pgmpy_cpds.append(pgmpy_cpd)

    # Add edges to pgmpy model
    for source, target in sorted(edges):
        pgmpy_pgm.add_edge(source, target)

    # Add converted CPDs
    for cpd in pgmpy_cpds:
        pgmpy_pgm.add_cpds(cpd)

    # Verify the model
    if check_model:
        pgmpy_pgm.check_model()

    return pgmpy_pgm


def _convert_conin_cpd_to_pgmpy_cpd(conin_cpd, conin_pgm):
    """
    Convert a single conin DiscreteCPD to pgmpy TabularCPD.

    Parameters
    ----------
    conin_cpd : DiscreteCPD
        The conin CPD to convert
    conin_pgm : DiscreteBayesianNetwork
        The conin model containing state information

    Returns
    -------
    pgmpy_TabularCPD
        The converted pgmpy CPD
    """
    # Get variable cardinality
    variable_states = conin_pgm.states_of(conin_cpd.node)
    variable_card = len(variable_states)

    #
    # Convert conin dict format to pgmpy 2D array format
    # conin: {0: [0.2, 0.8], 1: [0.9, 0.1]} -> pgmpy: [[0.2, 0.9], [0.8, 0.1]]
    #
    # Handle root nodes (no parents)
    if not conin_cpd.parents:
        values = [[conin_cpd.values[p]] for p in variable_states]

        return pgmpy_TabularCPD(
            variable=conin_cpd.node, variable_card=variable_card, values=values
        )

    # Handle nodes with parents
    else:
        values = [list() for _ in range(variable_card)]
        for p in _generate_parent_assignments(conin_cpd.parents, conin_pgm):
            for i, s in enumerate(variable_states):
                values[i].append(conin_cpd.values[p][s])

        # Get parent cardinalities
        evidence_card = [
            len(conin_pgm.states_of(parent)) for parent in conin_cpd.parents
        ]

        return pgmpy_TabularCPD(
            variable=conin_cpd.node,
            variable_card=variable_card,
            values=values,
            evidence=conin_cpd.parents,
            evidence_card=evidence_card,
        )


def _generate_parent_assignments(parents, conin_pgm):
    """
    Generate all possible parent assignments in the order expected by pgmpy.

    Parameters
    ----------
    parents : list
        List of parent node names
    conin_pgm : DiscreteBayesianNetwork
        The conin model containing state information

    Returns
    -------
    Generator
        Generator for a list of parent assignments in pgmpy order
    """
    parent_states = [conin_pgm.states_of(parent) for parent in parents]
    if len(parent_states) == 1:
        for val in parent_states[0]:
            yield val
    else:
        for val in itertools.product(*parent_states):
            yield val
