from conin.util import Util
from conin.dynamic_bayesian_network.expr import ExpressionNode


def get_cpd(G, node):
    '''
    Get the cpd for a node in G
    Note that this depends on the time index G.t
    '''
    if G.t.value() is None:
        raise Valueerror(f"G.t cannot be None")
    
    if node in G.dynamic_nodes:
        # for dynamic nodes we may have an alternate cpd if t==0
        potential_cpds = [
            _cpd
            for _cpd in self.G.cpds
            if (isinstance(_cpd.node, tuple) and _cpd.node[0] == node)
        ]
        if len(potential_cpds) == 0 or len(potential_cpds) > 2:
            raise ValueError(
                f"Unexpected no. of potential cpds ({len(potential_cpds)}) for dynamic node {node}"
            )
        if len(potential_cpds) > 1 and self.G.t.value() == 0:
            # this node has a (node, 0) initializer, and use it
            cpd = [_cpd for _cpd in potential_cpds if _cpd.node[1] == 0][0]
        elif len(potential_cpds) > 1 and self.G.t.value() > 0:
            # this node has a (node, 0) initializer, but don't use it
            cpd = [
                _cpd
                for _cpd in potential_cpds
                if isinstance(_cpd.node[1], ExpressionNode)
            ][0]
        else:
            # just one match
            cpd = potential_cpds[0]

    elif node in self.G.nodes:
        # static nodes only have one match
        potential_cpds = [_cpd for _cpd in self.G.cpds if _cpd.node == node]
        if len(potential_cpds) != 1:
            raise ValueError(
                f"Unexpected no. of potential cpds ({len(potential_cpds)}) for static node {node}"
            )
        cpd = potential_cpds[0]

    else:
        raise ValueError(f"Could not find node {node} in G.cpds")

    return cpd
    


def topological_sort(G):
    