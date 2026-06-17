from collections import deque
from conin.bayesian_network import DiscreteBayesianNetwork


def is_polytree(pgm):
    """
    Takes in a conin BN model and returns whether it is a polytree.
    If a BN is a polytree, then belief propogation is an exact algorithm.

    Does a BFS.

    Inputs:
        - pgm: conin discrete BN model
    Output:
        - bool
    """
    if not (type(pgm) is DiscreteBayesianNetwork):
        raise TypeError(
            f"is_polytree() expects an argument of type DiscreteBayesianNetwork ({type(pgm)=})"
        )

    nodes = set(pgm.nodes)
    nbrs = {node: set() for node in nodes}

    for cpd in pgm.cpds:
        if cpd.parents:
            for p in cpd.parents:
                nbrs[p].add(cpd.node)
                nbrs[cpd.node].add(p)

    return _is_polytree(nodes, nbrs)


def _is_polytree(nodes, nbrs):
    """
    This is the core logic for is_polytree, which is shared with other PGM functions.
    """

    visited = set()
    queue = deque()

    while True:
        if len(queue) == 0:
            if len(visited) == len(nodes):
                return True
            else:
                queue.append(next(iter(nodes - visited)))

        else:
            u = queue.popleft()

            if nbrs[u] & set(queue):
                return False

            visited.add(u)

            for v in nbrs[u]:
                if v not in visited:
                    queue.append(v)
