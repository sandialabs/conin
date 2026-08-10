from conin.common.conin.is_polytree import _is_polytree


def is_polytree(pgm):
    """
    Takes in a pgmpy model and returns whether it is a polytree.
    If a BN is a polytree, then belief propogation is an exact algorithm.

    Does a BFS.

    Inputs:
        - pgm: pgmpy model
    Output:
        - bool
    """
    nodes = set(pgm.nodes)
    nbrs = {node: set() for node in nodes}

    for u, nbr_set in pgm._adj.items():
        for v in nbr_set:
            nbrs[u].add(v)
            nbrs[v].add(u)

    return _is_polytree(nodes, nbrs)
