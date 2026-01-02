import itertools
from math import prod
from dataclasses import dataclass

from conin.util import try_import


@dataclass(slots=True)
class DiscreteFactor:
    """A factor over one or more discrete variables.

    Represents a non-negative function over random variables with finite state spaces. The
    ``values`` can be provided either as a mapping from assignments to
    weights or as a list of weights.  The :meth:`normalize` method is used to convert
    the factor representation to a dictionary keyed by assignments.

    :param list nodes: Ordered list of node identifiers that correspond to random variables over which the factor is defined.
    :param values: Factor weights as a dictionary or a list. For a
        dictionary, keys are node values. Use a single value for unary
        factors (e.g., ``{"A_val": weight}``) and tuples for multi-variate
        factors (e.g., ``{("A_val", "B_val"): weight}``). A list can be
        used to specify factor weights, where factor weights are associated with
        the Cartesian product of the node values.
    :type values: list | dict
    :param default_value: Optional fallback value that is used when a specific
        assignment is missing. This data is only used when factor weights are specified
        with a dictionary.  (default is 0)
    :type default_value: int | str
    """

    nodes: list
    values: list | dict
    default_value: str | int = 0

    def assignments(self, states):
        """Iterate over assignments for this factor's scope.

        Parameters
        ----------
        states : Mapping[Any, Iterable[Any]]
            Mapping from node identifiers to the ordered collection of allowed
            state values.

        Yields
        ------
        list of tuple
            A sequence of ``(node, value)`` pairs ordered consistently with
            :attr:`nodes`.
        """
        if len(self.nodes) == 1:
            for node_value in states[self.nodes[0]]:
                yield [(self.nodes[0], node_value)]
        else:
            slist = [states[node] for node in self.nodes]
            for assignment in itertools.product(*slist):
                yield [(node, assignment[i]) for i, node in enumerate(self.nodes)]

    def normalize(self, pgm):
        """Return a factor with dictionary-valued weights.

        If this factor's ``values`` are already a dictionary, ``self`` is
        returned. Otherwise, the flat list is converted to a dictionary keyed by
        assignments constructed from the provided model's state order.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork
            The probabilistic graphical model whose node state definitions are
            used to expand list-based factor values.

        Returns
        -------
        DiscreteFactor
            A factor whose ``values`` attribute is a dictionary keyed by state
            assignments instead of a flat list.
        """
        if type(self.values) is dict:
            return self
        else:
            slist = [pgm.states_of(node) for node in self.nodes]
            if len(slist) == 1:
                values = dict(zip(slist[0], self.values))
            else:
                indices = [
                    index[::-1] for index in itertools.product(*list(reversed(slist)))
                ]
                values = dict(zip(indices, self.values))

            return DiscreteFactor(
                nodes=self.nodes, values=values, default_value=self.default_value
            )


class DiscreteMarkovNetwork:
    """A discrete Markov network over finite-valued nodes.

    Stores node state spaces, optional undirected edges, and factor potentials.
    If edges are not explicitly provided, they are inferred from factor scopes.
    When factors are assigned, any list-based ``values`` are normalized to a
    dictionary keyed by assignments using the model's states.
    """

    """Markov network with discrete variables and factor potentials."""

    def __init__(self, *, states={}, edges=None, factors=[]):
        """Initialize a discrete Markov network.

        Parameters
        ----------
        states : dict or list, optional
            Mapping from node identifiers to ordered states or a list of node
            cardinalities. Defaults to an empty dictionary.

        edges : Iterable[tuple], optional
            Pairwise edges that connect nodes in the network.

        factors : Iterable[DiscreteFactor], optional
            Factors that encode the potential of each clique in the network.
        """
        self._nodes = []
        self._edges = edges
        self._factors = factors
        self.states = states

    def check_model(self):
        """Validate structural consistency of nodes, edges, and factors.

        Ensures that:

        - All nodes referenced by edges and factor scopes match the model's
          nodes.
        - Factor values are non-negative.
        - For list-based factors, the number of values equals the product of
          the cardinalities for the factor's nodes.

        Raises
        ------
        AssertionError
            If the states, edges, or factors are inconsistent with one
            another, or if any factor value is negative.
        """
        model_nodes = set(self._states.keys())

        if self._edges:
            enodes = set()
            for v, w in self._edges:
                enodes.add(v)
                enodes.add(w)

            # Note: We have an error if the edges contain nodes that aren't in the model.
            # BUT, we may have a model_node that is not in any edge
            if len(enodes - model_nodes) > 0:
                raise RuntimeError(
                    f"Unexpected discrepancy in model nodes: the set of edge nodes contains the following nodes that do not appear in the model: {enodes-model_nodes}"
                )

        fnodes = set()
        for factor_num, f in enumerate(self._factors):
            for node in f.nodes:
                fnodes.add(node)
            if type(f.values) is dict:
                for k, v in f.values.items():
                    assert v >= 0, f"Unexpected negative factor value {v}"
                    if type(k) is tuple:
                        for i, iv in enumerate(k):
                            assert (
                                iv in self._states[f.nodes[i]]
                            ), f"Unexpected node value {k} for factor {factor_num}: the {i}-th value is {iv} but should be  in {self._states[f.nodes[i]]}"
                    else:
                        assert (
                            k in self._states[f.nodes[0]]
                        ), f"Unexpected node value {k} in factor {factor_num}"
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
        """List of node identifiers in the model.

        Returns
        -------
        list
            Node identifiers registered with the model.
        """
        return self._nodes

    @property
    def states(self):
        """Mapping from node to its allowed states.

        Returns
        -------
        dict
            Dictionary mapping each node to the ordered list of valid states.
        """
        return self._states

    @states.setter
    def states(self, values):
        """Set node states from cardinalities or explicit values.

        If a list is provided, nodes are created as integers ``0..n-1`` and
        each entry indicates a node's cardinality (values ``0..card-1``). If a
        dictionary is provided, keys are node identifiers and values are the
        explicit lists of allowed states.

        Parameters
        ----------
        values : list of int or dict
            Either a list containing the cardinality for each anonymous node or
            a dictionary that directly maps node identifiers to ordered states.

        Raises
        ------
        TypeError
            If ``values`` is neither a list nor a dictionary.

        Examples
        --------
        >>> dmn = DiscreteMarkovNetwork()
        >>> dmn.states = [4, 3]
        >>> dmn.nodes
        [0, 1]
        >>> dmn.states
        {0: [0, 1, 2, 3], 1: [0, 1, 2]}

        >>> dmn = DiscreteMarkovNetwork()
        >>> dmn.states = {"A": ["T", "F"], "B": [-1, 1]}
        >>> dmn.nodes
        ['A', 'B']
        >>> dmn.states
        {'A': ['T', 'F'], 'B': [-1, 1]}
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
        """Return the allowed states for a node.

        Parameters
        ----------
        node : Hashable
            Identifier of the node of interest.

        Returns
        -------
        list
            Ordered list of valid states for ``node``.
        """
        return self._states[node]

    def card(self, node):
        """Return the cardinality of a node.

        Parameters
        ----------
        node : Hashable
            Identifier of the node of interest.

        Returns
        -------
        int
            Cardinality of the node.
        """
        return len(self._states[node])

    #
    # Edges
    #

    @property
    def edges(self):
        """List of undirected edges.

        If edges are not explicitly set, they are inferred from the scopes of
        the current factors.

        Returns
        -------
        list
            Edge list describing pairwise connections between nodes.
        """
        if not self._edges:
            edges = set()
            for factor in self._factors:
                for edge in itertools.combinations(sorted(factor.nodes), 2):
                    edges.add(edge)
            self._edges = list(edges)
        return self._edges

    @edges.setter
    def edges(self, edges):
        """Set the list of undirected edges.

        Parameters
        ----------
        edges : Iterable[tuple]
            Collection of pairwise node connections.

        Examples
        --------
        >>> dmn = DiscreteMarkovNetwork()
        >>> dmn.states = [4, 3]
        >>> dmn.edges = [(0, 1), (1, 2)]
        """
        self._edges = list(edges)

    #
    # Factors
    #

    @property
    def factors(self):
        """Return the factors registered with the model.

        Returns
        -------
        list of DiscreteFactor
            Factor objects representing clique potentials.
        """
        return self._factors

    @factors.setter
    def factors(self, factor_list):
        """Attach a list of factors to the model.

        Each factor is converted to the dictionary form via
        :meth:`DiscreteFactor.normalize` using the current model states.

        Parameters
        ----------
        factor_list : Iterable[DiscreteFactor]
            Factors that define the potential of each clique. Factors with
            list-valued potentials are normalized against the current state
            definitions.

        Examples
        --------
        >>> dmn = DiscreteMarkovNetwork()
        >>> dmn.states = [4, 3]
        >>> f1 = DiscreteFactor(nodes=[0, 1], values={(0, 0): 1, (0, 1): 2, (0, 2): 3})
        >>> f2 = DiscreteFactor(nodes=[0], values={0: 0, 1: 1, 2: 2, 3: 3})
        >>> dmn.factors = [f1, f2]
        """
        self._factors = [factor.normalize(self) for factor in factor_list]

    def num_factor_parameters(self):
        return sum(len(f.values) for f in self.factors)


class ConstrainedDiscreteMarkovNetwork:
    """Markov network that supports custom constraints.

    Wraps a :class:`DiscreteMarkovNetwork` and an optional constraint functor
    that decorates a Pyomo model with additional feasibility constraints.
    """

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained wrapper.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork
            Underlying probabilistic graphical model to constrain.
        constraints : Callable or None, optional
            Functor that applies additional restrictions during inference.
        """
        self.pgm = pgm
        if constraints:
            self._constraints = constraints
        else:
            self._constraints = []

    def check_model(self):
        """Validate the underlying model."""
        self.pgm.check_model()

    @property
    def nodes(self):
        """Return the underlying model's nodes.

        Returns
        -------
        list
            Node identifiers maintained by the underlying model.
        """
        return self.pgm.nodes

    def states_of(self, node):
        """Return the allowed states for a node.

        Parameters
        ----------
        node : Hashable
            Identifier of the node of interest.

        Returns
        -------
        list
            Ordered list of valid states for ``node``.
        """
        return self.pgm.states_of(node)

    @property
    def constraints(self):
        """Get a list of constraint functors.

        :return: The constraint functor or ``None`` if not set.
        :rtype: callable | None
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set a list of functions that are used to define model constraints.

        Parameters
        ----------
        constraint_list : List[Callable]
            List of functions that generate model constraints.
        """
        assert type(constraint_list) is list
        self._constraints = constraint_list
