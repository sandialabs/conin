import itertools
from math import prod
from dataclasses import dataclass

from conin.util import try_import
from conin.markov_network.inference import create_MN_map_query_model

with try_import() as pgmpy_available:
    import pgmpy.models


@dataclass(slots=True)
class DiscreteFactor:
    nodes: list
    values: list | dict
    default_value: str | int = 0

    def assignments(self, states):
        """
        Each assignment is a list of (node,value) pairs, in the same order as self.nodes.
        """
        if len(self.nodes) == 1:
            for node_value in states[self.nodes[0]]:
                yield [(self.nodes[0], node_value)]
        else:
            slist = [states[node] for node in self.nodes]
            for assignment in itertools.product(*slist):
                yield [(node, assignment[i]) for i, node in enumerate(self.nodes)]

    def normalize(self, pgm):
        if type(self.values) is dict:
            return self
        else:
            slist = [pgm.states_of(node) for node in self.nodes]
            if len(slist) == 1:
                values = dict(zip(slist[0], self.values))
            else:
                values = dict(zip(itertools.product(*slist), self.values))
            return DiscreteFactor(
                nodes=self.nodes, values=values, default_value=self.default_value
            )


class DiscreteMarkovNetwork:

    def __init__(self, *, states={}, edges=None, factors=[]):
        self._nodes = []
        self._edges = edges
        self._factors = factors
        self.states = states

    def check_model(self):
        model_nodes = set(self._states.keys())

        if self._edges:
            enodes = set()
            for v, w in self._edges:
                enodes.add(v)
                enodes.add(w)

            # Note: We assert equality to ensure that all nodes are used in the model
            assert model_nodes == enodes

        fnodes = set()
        for f in self._factors:
            for node in f.nodes:
                fnodes.add(node)
            if type(f.values) is dict:
                for k, v in f.values.items():
                    assert v >= 0, f"Unexpected factor value {v}"
                    if type(k) is tuple:
                        for i, iv in enumerate(k):
                            assert (
                                iv in self._states[self._nodes[i]]
                            ), f"Unexpected value {iv} in the {i}-th node value of {k}"
                    else:
                        assert (
                            k in self._states[self._nodes[0]]
                        ), f"Unexpected node value {k}"
            else:
                assert (
                    self.states
                ), "The states attributes must be specified when the factor values are not a dictionary"
                for v in f.values:
                    assert v >= 0
                # We assert equality to ensure the list of values covers all combinations of
                # node states
                nodeset = set(f.nodes)
                assert len(f.values) == prod(
                    len(v) if k in nodeset else 1 for k, v in self.states.items()
                )

        # Note: We assert equality to ensure that all nodes are used in the model
        assert model_nodes == fnodes

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
        DMN = DiscreteMarkovNetwork()
        DMN.states = [4, 3]  # Cardinality of nodes
        assert DMN.nodes == [0,1]
        assert DMN.states == {0: [0,1,2,3], 1:[0,1,2]}

        DMN = DiscreteMarkovNetwork()
        DMN.states = {"A": ["T", "F"], "B": [-1, 1]}
        assert DMN.nodes == ["A", "B"]
        assert DMN.states == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._states = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._states = values

        else:
            raise TypeError(f"Unexpected type for states: {type(values)}")

    def states_of(self, node):
        return self._states[node]

    def card(self, node):
        return len(self._states[node])

    #
    # Edges
    #

    @property
    def edges(self):
        if not self._edges:
            edges = set()
            for factor in self._factors:
                for edge in itertools.combinations(sorted(factor.nodes), 2):
                    edges.add(edge)
            self._edges = list(edges)
        return self._edges

    @edges.setter
    def edges(self, edges):
        """
        DMN = DiscreteMarkovNetwork()
        DMN.states = [4, 3]  # Cardinality of nodes
        DMN.edges = [ (0,1), (1,2) ]
        """
        self._edges = list(edges)

    #
    # Factors
    #

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, factor_list):
        """
        DMN = DiscreteMarkovNetwork()
        DMN.states = [4, 3]  # Cardinality of nodes
        f1 = DiscreteFactor(nodes=[0,1], values={(0,0):1, (0,1):2, (0,2):3})
        f2 = DiscreteFactor(nodes=[0], values={0:0, 1:1, 2:2, 3:3})
        DMN.factors = [f1, f2]
        """
        self._factors = [factor.normalize(self) for factor in factor_list]

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        return create_MN_map_query_model(
            pgm=self,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )


class ConstrainedDiscreteMarkovNetwork:

    def __init__(self, pgm, constraints=None):
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes

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
