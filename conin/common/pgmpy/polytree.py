from collections import deque


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
    nodes = pgm.nodes
    nbrs = {node: set() for node in nodes}

    for u, nbr_set in pgm._adj.items():
        for v in nbr_set:
            nbrs[u].add(v)
            nbrs[v].add(u)

    visited = set()
    queue = deque()

    while True:
        if len(queue) == 0:
            if len(visited) == len(nodes):
                return True
            else:
                # WEH - Do we just need to pick any member of the set: set(nodes) - visited?
                queue.append(list(set(nodes) - visited)[0])

        else:
            u = queue.popleft()

            if nbrs[u] & set(queue):
                return False

            visited.add(u)

            for v in nbrs[u]:
                if v not in visited:
                    queue.append(v)
