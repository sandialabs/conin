import itertools
from math import prod
from dataclasses import dataclass

from conin.util import try_import
from conin.markov_network.inference import create_MN_map_query_model

with try_import() as pgmpy_available:
    import pgmpy.models


#
# NOTE: The states dictionary is needed to generate assignments, but not for other
# factor operations.  We leaves this off the factor object for now, until we know that we
# need a per-factor states dictionary.
#
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


class DiscreteMarkovNetwork:

    def __init__(self, *, states={}, edges=None, factors=[]):
        self._nodes = []
        self._states = states
        self._edges = edges if edges else []
        self._factors = factors

    def check_model(self):
        model_nodes = set(self._states.keys())

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
                assert len(f.values) == prod(len(v) for _, v in self.states)

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
        self._factors = factor_list


class ConstrainedDiscreteMarkovNetwork:

    def __init__(self, pgm, constraints=None):
        self.pgm = convert_to_DiscreteMarkovNetwork(pgm)
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


def get_factor_value(f, values):
    """
    This is a wrapper for the DiscreteFactor.get_value() method.  This allows for
    the specification of node names that are non-strings.
    """
    return f.values[tuple(f.name_to_no[var][values[var]] for var in f.variables)]


class PgmpyWrapperDiscreteMarkovNetwork(DiscreteMarkovNetwork):

    def __init__(self, pgmpy_pgm):
        super().__init__()
        self._pgmpy_pgm = pgmpy_pgm
        self.states = pgmpy_pgm.states

        factors = []
        for factor in pgmpy_pgm.get_factors():
            vars = factor.scope()
            size = prod(factor.get_cardinality(vars).values())

            if len(vars) == 1:
                values = {
                    key[0][1]: get_factor_value(factor, dict(key))
                    for key in factor.assignment(list(range(size)))
                }
            else:
                values = {
                    tuple(v for _, v in key): get_factor_value(factor, dict(key))
                    for key in factor.assignment(list(range(size)))
                }

            factors.append(DiscreteFactor(nodes=factor.variables, values=values))
        self.factors = factors


def convert_to_DiscreteMarkovNetwork(pgm):
    if (
        type(pgm) is DiscreteMarkovNetwork
        or type(pgm) is PgmpyWrapperDiscreteMarkovNetwork
    ):
        return pgm

    elif type(pgm) is pgmpy.models.MarkovNetwork:
        return PgmpyWrapperDiscreteMarkovNetwork(pgm)

    else:
        raise TypeError(f"Unexpected markov network type: {type(pgm)}")

