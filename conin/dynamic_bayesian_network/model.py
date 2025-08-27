import munch

from conin.dynamic_bayesian_network.inference import create_DDBN_map_query_model
from conin.dynamic_bayesian_network.expr import ExpressionVariable


class DynamicDiscreteBayesianNetwork:

    def __init__(self, *, states={}, dynamic_states={}, cpds=[]):
        self._nodes = []
        self._dynamic_nodes = []
        self._edges = []
        self._dynamic_edges = []
        self._states = states
        self._dynamic_states = dynamic_states
        self._cpds = cpds
        self.t = ExpressionVariable()

    def check_model(self):
        return True

    #
    # Nodes
    #

    @property
    def nodes(self):
        return self._nodes

    @property
    def dynamic_nodes(self):
        return self._dynamic_nodes

    #
    # Edges
    #

    @property
    def edges(self):
        return self._edges

    @property
    def dynamic_edges(self):
        return self._dynamic_edges

    #
    # States
    #

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        """
        DDBN = DynamicDiscreteBayesianNetwork()
        DDBN.states = [4, 3]  # Cardinality of nodes
        assert DDBN.nodes == [0,1]
        assert DDBN.states == {0: [0,1,2,3], 1:[0,1,2]}

        DDBN = DynamicDiscreteBayesianNetwork()
        DDBN.states = {"A": ["T", "F"], "B": [-1, 1]}
        assert DDBN.nodes == ["A", "B"]
        assert DDBN.states == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._states = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._states = values

        else:
            raise TypeError(f"Unexpected type for states: {type(values)}")

    @property
    def dynamic_states(self):
        return self._dynamic_states

    @dynamic_states.setter
    def dynamic_states(self, values):
        """
        DDBN = DynamicDiscreteBayesianNetwork()
        DDBN.dynamic_states = {"A": ["T", "F"], "B": [-1, 1]}
        assert DBN.dynamic_nodes == ["A", "B"]
        assert DBN.dynamic_states == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is dict:
            self._dynamic_nodes = sorted(values.keys())
            self._dynamic_states = values

        else:
            raise TypeError(f"Unexpected type for dynamic states: {type(values)}")

    def states_of(self, node):
        if node in self._states:
            return self._states[node]
        elif node in self._dynamic_states:
            return self._dynamic_states[node]
        elif type(node) is tuple and node[0] in self._dynamic_states:
            return self._dynamic_states[node[0]]
        raise ValueError(f"Unexpected node value: {node}")

    def card(self, node):
        if node in self._states:
            return len(self._states[node])
        elif node in self._dynamic_states:
            return len(self._dynamic_states[node])
        elif type(node) is tuple and node[0] in self._dynamic_states:
            return len(self._dynamic_states[node[0]])
        raise ValueError(f"Unexpected node value: {node}")

    #
    # CPDs
    #

    @property
    def cpds(self):
        return self._cpds

    @cpds.setter
    def cpds(self, cpd_list):
        """
        DBN = DiscreteBayesianNetwork()
        DDBN.states = {"X": ["T", "F"], "Y": [-1, 1]}
        DDBN.dynamic_states = {"A": ["t", "f"], "B": [2, 3]}

        # A CPD for ("A",t) depending on "X", for all t
        c1 = DiscreteCPD(variable=("A",None), evidence=["X"],
                values={"T": dict("t": 0.3, "f": 0.7),
                        "F": dict("t": 0.4, "f": 0.6)})

        # A CPD for ("B",0) depending on "X", for t==0
        c2 = DiscreteCPD(variable=("B",0), evidence=["Y"],
                values={"T": dict(2: 0.3, 3: 0.7},
                        "F": dict(2: 0.4, 3: 0.6)})

        # A CPD for ("B",t) depending on ("A",t), for all t
        c3 = DiscreteCPD(variable=("B",None), evidence=[("A",0)],
                values={"t": dict(2: 0.3, 3: 0.7},
                        "f": dict(2: 0.4, 3: 0.6)})

        # A CPD for ("A",t) depending on ("A",t-1), for all t
        c4 = DiscreteCPD(variable=("A",None), evidence=[("A",-1)],
                values={"t": dict("t": 0.3, "f": 0.7},
                        "f": dict("t": 0.4, "f": 0.6)})

        DMN.cpds = [c1, c2]
        """
        self._cpds = [cpd.normalize(self) for cpd in cpd_list]

    def create_map_query_model(self, *, start=0, stop=1, variables=None, evidence=None):
        return create_DDBN_map_query_model(
            pgm=self,
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
        )


class ConstrainedDynamicDiscreteBayesianNetwork:

    def __init__(self, pgm, constraints=None):
        if isinstance(pgm, DynamicDiscreteBayesianNetwork):
            self.pgm = pgm
        else:
            from conin.dynamic_bayesian_network.model_pgmpy import (
                convert_to_DynamicDiscreteBayesianNetwork,
            )

            self.pgm = convert_to_DynamicDiscreteBayesianNetwork(pgm)
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model, data):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model, data)
        return model

    def create_map_query_model(self, *, start=0, stop=1, variables=None, evidence=None):
        model = create_DDBN_map_query_model(
            pgm=self.pgm,
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
        )
        self.data = munch.Munch(
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            T=list(range(start, stop + 1)),
        )
        return self.create_constraints(model, self.data)
