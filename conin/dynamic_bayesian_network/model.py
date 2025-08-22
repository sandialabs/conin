import munch

#from conin.util import try_import
from conin.dynamic_bayesian_network.inference import create_DDBN_map_query_model

#with try_import() as pgmpy_available:
#    import pgmpy.models


class DynamicDiscreteBayesianNetwork:

    def __init__(self, *, states={}, edges=None, cpds=[]):
        self._nodes = []
        self._states = states
        self._edges = edges if edges else []
        self._cpds = cpds

    #
    # Nodes
    #

    @property
    def nodes(self):
        return self._nodes

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        """
        DBN = DiscreteBayesianNetwork()
        DBN.states = [4, 3]  # Cardinality of nodes
        assert DBN.nodes == [0,1]
        assert DBN.states == {0: [0,1,2,3], 1:[0,1,2]}

        DBN = DiscreteBayesianNetwork()
        DBN.states = {"A": ["T", "F"], "B": [-1, 1]}
        assert DBN.nodes == ["A", "B"]
        assert DBN.states == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._states = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._states = values

        else:
            raise TypeError(f"Unexpected type for states: {type(values)}")

    def card(self, node):
        return len(self._states[node])

    #
    # Edges
    #

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        """
        DBN = DiscreteBayesianNetwork()
        DBN.states = [4, 3]  # Cardinality of nodes
        DBN.edges = [ (0,1), (1,2) ]
        """
        self._edges = list(edges)

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
        DBN.states = [4, 3]  # Cardinality of nodes
        c1 = TabularCPD(nodes=[0,1], values={(0,0):1, (0,1):2, (0,2):3})
        c2 = TabularCPD(nodes=[0], values={0:0, 1:1, 2:2, 3:3})
        DMN.cpds = [c1, c2]
        """
        self._cpds = cpd_list

    def create_map_query_model(self, *, start=0, stop=1, variables=None, evidence=None):
        return create_DDBN_map_query_model(
            pgm=self.pgm,
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
        )


class ConstrainedDynamicDiscreteBayesianNetwork:

    def __init__(self, pgm, constraints=None):
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


def convert_to_DynamicDiscreteBayesianNetwork(pgm):
    return pgm
