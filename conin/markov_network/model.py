import itertools
from math import prod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiscreteFactor:
    """A factor over one or more discrete variables.

    Represents a non-negative function over random variables with finite state
    spaces. The ``values`` can be provided either as a mapping from assignments
    to weights or as a list of weights. The ``normalize`` method converts the
    factor representation to a dictionary keyed by assignments.

    Parameters
    ----------
    nodes : list
        Ordered list of node identifiers that correspond to random variables
        over which the factor is defined.
    values : list or dict
        Factor weights as a dictionary or a list. For a dictionary, keys are
        node values. Use a single value for unary factors, such as
        ``{"A_val": weight}``, and tuples for multivariate factors, such as
        ``{("A_val", "B_val"): weight}``. A list can be used to specify
        factor weights associated with the Cartesian product of the node
        values.
    default_value : int or str, optional
        Optional fallback value used when a specific assignment is missing.
        This value is only used when factor weights are specified with a
        dictionary. Defaults to 0.
    """

    nodes: list
    values: list | dict
    default_value: str | int = 0

    def __post_init__(self):
        object.__setattr__(
            self,
            "values",
            dict(self.values) if isinstance(self.values, dict) else list(self.values),
        )
        object.__setattr__(self, "nodes", list(self.nodes))

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
            ``nodes``.
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
        if isinstance(states, list):
            self._nodes = list(range(len(states)))
            self._states = {i: list(range(v)) for i, v in enumerate(states)}
        elif isinstance(states, dict):
            tmp = list(states.keys())
            tmp.sort(key=lambda x: (x,) if not isinstance(x, tuple) else x)
            self._nodes = tmp
            self._states = dict(states)
        else:
            raise TypeError(f"Unexpected type for states: {type(states)}")

        self._edges = list(edges) if edges is not None else None

        tmp = set(self._nodes)
        for factor in factors:
            for node in factor.nodes:
                if node not in tmp:
                    tmp.add(node)
                    self._nodes.append(node)
        self._factors = [factor.normalize(self) for factor in factors]

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
        RuntimeError
            If an explicitly provided edge references a node that does not
            appear in the model.
        AssertionError
            If the states, edges, or factors are inconsistent with one another,
            if any factor value is negative, or if a factor uses an invalid
            node state.
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
        return dict(self._states)

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
        return list(self._edges)

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
        return list(self._factors)

    def with_factor(self, factor):
        """Return a new model with ``factor`` appended to its factor list.

        Parameters
        ----------
        factor : DiscreteFactor
            Factor to add to the model.

        Returns
        -------
        DiscreteMarkovNetwork
            A new model with the same states and edges as ``self`` plus the
            additional factor.

        Examples
        --------
        >>> from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor
        >>> dmn = DiscreteMarkovNetwork(states=[4, 3])
        >>> f1 = DiscreteFactor(nodes=[0, 1], values={(0, 0): 1, (0, 1): 2, (0, 2): 3})
        >>> f2 = DiscreteFactor(nodes=[0], values={0: 0, 1: 1, 2: 2, 3: 3})
        >>> dmn = dmn.with_factor(f1).with_factor(f2)
        """
        return DiscreteMarkovNetwork(
            states=self.states, edges=self._edges, factors=self.factors + [factor]
        )

    def num_factor_parameters(self):
        return sum(len(f.values) for f in self.factors)


class ConstrainedDiscreteMarkovNetwork:
    """Markov network that supports custom constraints.

    Wraps a ``DiscreteMarkovNetwork`` and optional ``ConstraintFunctor``
    instances that apply additional feasibility constraints.
    """

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained wrapper.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork
            Underlying probabilistic graphical model to constrain.
        constraints : list[ConstraintFunctor] or None, optional
            Constraint functors that apply additional restrictions during
            inference.
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
        """Return the configured constraint functors.

        Returns
        -------
        list
            List of ``ConstraintFunctor`` instances that add constraints to an
            inference model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set the functions used to define model constraints.

        Parameters
        ----------
        constraint_list : list of ConstraintFunctor
            List of ``ConstraintFunctor`` instances that generate model
            constraints.
        """
        assert type(constraint_list) is list
        self._constraints = constraint_list
