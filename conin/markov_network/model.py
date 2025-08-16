try:
    import pgmpy.models

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False

from conin.markov_network.inference import create_MN_map_query_model


class DiscreteMarkovNetwork:

    def __init__(self, *, node_values={}, edges=[], factors=[]):
        self.node_values = node_values
        self.edges = edges
        self.factors = factors

    def check_model(self):
        nodes = set(self._nodes)

        enodes = set()
        for v, w in self._edges:
            enodes.add(v)
            enodes.add(w)

        assert nodes == enodes

        fnodes = set()
        for nodes, values in self._factors:
            for node in nodes:
                fnodes.add(node)
            nset = set(nodes)
            for k, v in values:
                assert v >= 0, f"Unexpected factor value {v}"
                # TODO - test variables
                # TODO - test variable values

    #
    # Nodes
    #

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_values(self):
        return self._node_values

    @node_values.setter
    def node_values(self, values):
        """
        DMN = DiscreteMarkovNetwork()
        DMN.node_values = [4, 3]  # Cardinality of nodes
        assert DMN.nodes == [0,1]
        assert DMN.node_values == {0: [0,1,2,3], 1:[0,1,2]}

        DMN = DiscreteMarkovNetwork()
        DMN.node_values = {"A": ["T", "F"], "B": [-1, 1]}
        assert DMN.nodes == ["A", "B"]
        assert DMN.node_values == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._node_values = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._node_values = values

    def card(self, node):
        return len(self._node_values[node])

    #
    # Edges
    #

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, values):
        self._edges = values

    def add_edges(self, *args):
        self._edges.extend(args)

    #
    # Factors
    #

    def add_factor(self, nodes, values):
        self._factors.append((nodes, values))


class ConstrainedMarkovNetwork:

    def __init__(self, pgm, constraints=None):
        assert pgmpy_available and isinstance(
            pgm, pgmpy.models.MarkovNetwork
        ), "Argument must be a pgmpy MarkovNetwork"
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model)
        return model

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        model = create_MN_map_query_model(
            pgm=self.pgm,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )
        return self.create_constraints(model)
