import itertools
from conin.util import try_import
from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor

with try_import() as pgmpy_available:
    from pgmpy.models import DiscreteMarkovNetwork as pgmpy_DiscreteMarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor as pgmpy_DiscreteFactor


def convert_conin_to_pgmpy_mn(conin_pgm, check_model=True):
    """
    Convert a conin DiscreteMarkovNetwork to a pgmpy DiscreteMarkovNetwork.

    Parameters
    ----------
    conin_pgm : DiscreteMarkovNetwork
        The conin Markov network to convert

    Returns
    -------
    pgmpy_DiscreteMarkovNetwork
        The converted pgmpy Markov network

    Raises
    ------
    ImportError
        If pgmpy is not available
    ValueError
        If the input is not a conin DiscreteMarkovNetwork
    """
    if not pgmpy_available:
        raise ImportError("Cannot convert to pgmpy without pgmpy installed")

    if not isinstance(conin_pgm, DiscreteMarkovNetwork):
        raise ValueError(
            f"Expected conin DiscreteMarkovNetwork, got {type(conin_pgm)}"
        )

    # Create pgmpy model
    pgmpy_pgm = pgmpy_DiscreteMarkovNetwork()

    # Add nodes
    for node in conin_pgm.nodes:
        pgmpy_pgm.add_node(node)

    # Convert factors and build edges
    pgmpy_factors = []
    edges = set()

    for conin_factor in conin_pgm.factors:
        # Convert factor
        #import pprint
        #print("")
        #pprint.pprint(conin_factor.values)
        #pprint.pprint(conin_pgm.states)
        #pprint.pprint( list(conin_factor.assignments(conin_pgm.states) ))
        if len(conin_factor.nodes) == 1:
            values = [conin_factor.values[v] for v in conin_pgm.states[conin_factor.nodes[0]]]
        else:
            slist = [conin_pgm.states[node] for node in conin_factor.nodes]
            values = [conin_factor.values[assignment] for assignment in itertools.product(*slist)]

        pgmpy_factor = pgmpy_DiscreteFactor(variables=conin_factor.nodes, 
                                cardinality=[len(conin_pgm.states[v]) for v in conin_factor.nodes],
                                values=values)

        # Track edges for parent-child relationships
        #if conin_factor.parents:
        #    for parent in conin_factor.parents:
        #        edges.add((parent, conin_factor.node))

        pgmpy_factors.append(pgmpy_factor)

    # Add edges to pgmpy model
    pgmpy_pgm.add_edges_from( conin_pgm.edges )

    # Add converted factors
    for factor in pgmpy_factors:
        pgmpy_pgm.add_factors(factor)

    # Verify the model
    if check_model:
        pgmpy_pgm.check_model()

    return pgmpy_pgm

