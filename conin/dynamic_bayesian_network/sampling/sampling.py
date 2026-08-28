from conin.util import Util
from conin.dynamic_bayesian_network.expr import ExpressionNode


class Sampler:
    '''
    Class to handle sampling functions for the DBN G
    '''
    def __init__(self, G):
        self.G = G
        self.G.t.set_value(0)
        self.cpd_map = self.get_cpd_map()
        self.representation = self.build_representation()

    def set_time(self, t):
        rebuild = (self.G.t.value()==0) or (t==0)  # flag to rebuild representation
        self.G.t.set_value(t)
        if rebuild:
            self.cpd_map = self.get_cpd_map()
            self.representation = self.build_representation()

    def get_cpd(self, node):
        '''
        Get cpd for node in G
        '''    
        if node in self.G.dynamic_nodes:
            # for dynamic nodes we may have an alternate cpd if t==0
            potential_cpds = [
                _cpd for _cpd in self.G.cpds if (isinstance(_cpd.node, tuple) and _cpd.node[0]==node)
            ]
            if len(potential_cpds)==0 or len(potential_cpds)>2:
                raise ValueError(
                    f"Unexpected no. of potential cpds ({len(potential_cpds)}) for dynamic node {node}"
                )
            if len(potential_cpds)>1 and self.G.t.value()==0:
                # this node has a (node, 0) initializer, and use it
                cpd = [_cpd for _cpd in potential_cpds if _cpd.node[1]==0][0]
            elif len(potential_cpds)>1 and self.G.t.value()>0:
                # this node has a (node, 0) initializer, but don't use it
                cpd = [_cpd for _cpd in potential_cpds if isinstance(_cpd.node[1], ExpressionNode)][0]
            else:
                # just one match
                cpd = potential_cpds[0]
        
        elif node in self.G.nodes:
            # static nodes only have one match
            potential_cpds = [
                _cpd for _cpd in self.G.cpds if _cpd.node==node
            ]
            if len(potential_cpds)!=1:
                raise ValueError(
                    f"Unexpected no. of potential cpds ({len(potential_cpds)}) for static node {node}"
                )
            cpd = potential_cpds[0]
    
        else:
            raise ValueError(f"Could not find node {node} in G.cpds")
    
        return cpd

    def get_cpd_map(self):
        '''
        Get {node:cpd} mapping for G
        '''
        return {
            _node:self.get_cpd(_node) for _node in self.G.nodes + self.G.dynamic_nodes
        }

    def extract_parents(self, cpd):
        '''
        Build [(_node, previous_flag), ...] representation of parents in the given cpd in G
        '''
        return [
            (_parent[0], _parent[1].value()==(self.G.t.value() - 1)) if isinstance(_parent, tuple) else (  # dynamic
                (_parent, True)  # static
            ) for _parent in (cpd.parents or [])
        ]

    def get_children(self, node):
        '''
        Get children of the current node from the cpd_map
        A child is only considered *for the current time step*
        '''
        return [
            _node for (_node, _cpd) in self.cpd_map.items() for _parent in (_cpd.parents or [])
            if (
                (_parent[0]==node and _parent[1].value()==self.G.t.value()) if isinstance(_parent, tuple)
                else _parent==node
            )
        ]

    def build_representation(self):
        '''
        Build dictionary representation of nodes in G, i.e.
        {
          parents:   [(_node, previous_flag), ...]
          values:    [value_tuple:probabilty_list, ...]
          children:  [_node, ...]
        }
        '''    
        return {
            _node: {
                'parents':self.extract_parents(_cpd),
                'values':_cpd.values,
                'children':self.get_children(_node)
            } 
            for (_node, _cpd) in self.cpd_map.items()
        }
    
    def sample_next_states(self, previous_states={}):
        '''
        Sample the next states for the DBN G given its previous state data and
    
        The states are sampled using a topological sort (Kahn's algorithm),
          whereby sort nodes by their in-degrees in the DBN and only
          process nodes with in-degree 0, updating the in-degree as we go
        '''
        if self.G.t.value()==0 and len(previous_states)>0:
            raise ValueError('G.t==0 but previous_states is non-empty')
        
        # get all nodes that need to be updated
        nodes_to_update = self.G.dynamic_nodes.copy()  # prevent side-effects
        if self.G.t.value() == 0:  # initialize
            nodes_to_update += self.G.nodes
    
        # get mapping of each node to its in-degree
        degree_map = {
            _node:len(self.representation[_node]['parents']) for _node in nodes_to_update
        }
        
        # update the degrees of children of previous_states
        for _node in nodes_to_update:
            for _parent in self.representation[_node]['parents']:
                if _parent[1] and _parent[0] in previous_states:
                    degree_map[_node] -= 1
    
        # build queue of nodes with degree 0
        queue = [_node for _node in degree_map if degree_map[_node]==0]
    
        # process via topologial sort on the degrees
        next_states = {}    
        while len(queue)>0:    
            node = queue[0] 
    
            # get the parents and associated state values for this node
            parents = self.representation[node]['parents']            
            if len(parents)==0:
                # this node has a single values row
                row = self.representation[node]['values']
            else:
                # get the corresponding values from the parents
                table_key = tuple([
                    previous_states[_node] if _previous else next_states[_node] for (
                        _node, _previous) in parents
                ])
                row = self.representation[node]['values'][table_key]
    
            # sample a new value
            idx = Util.sample_from_vec(list(row.values()))
            next_states[node] = list(row.keys())[idx]
    
            # decrement node and children
            degree_map[node] -= 1
            for _child in self.representation[node]['children']:
                degree_map[_child] -= 1
    
            # update queue
            queue = [_node for _node in degree_map if degree_map[_node]==0]
    
        # any prevous_states that are not updated are copied to next_states
        for node in previous_states:
            if not node in next_states:
                next_states[node] = previous_states[node]
    
        return next_states

    def sample(self, T=10, N=10):
        '''
        Sample N traces of G, each with length T
        '''
        runs = {}
        for n in range(N):
            trace = []
            previous_states = {}
            for t in range(T):
                self.set_time(t)
                next_states = self.sample_next_states(previous_states=previous_states)
                trace.append(next_states)
                previous_states = next_states
            runs[n] = trace
        return runs
                

