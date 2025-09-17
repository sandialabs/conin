import itertools
from math import prod
from dataclasses import dataclass

from conin.util import try_import
from conin.markov_network.inference import create_MN_map_query_model


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

        Generates each combination of node values and yields it as a list of
        ``(node, value)`` pairs in the same order as :pyattr:`self.nodes`.

        :param dict states: Mapping from node identifier to a list of allowed
            values (i.e., ``{node: [v0, v1, ...]}``).
        :yields: Lists of ``(node, value)`` pairs describing an assignment.
        :rtype: Iterator[list[tuple[object, object]]]
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

        :param DiscreteMarkovNetwork pgm: A model that defines the states and
            their order for each node in this factor's scope.
        :return: A new factor whose ``values`` are a dictionary keyed by
            assignments.
        :rtype: DiscreteFactor
        """
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
    """A discrete Markov network over finite-valued nodes.

    Stores node state spaces, optional undirected edges, and factor potentials.
    If edges are not explicitly provided, they are inferred from factor scopes.
    When factors are assigned, any list-based ``values`` are normalized to a
    dictionary keyed by assignments using the model's states.
    """

    def __init__(self, *, states={}, edges=None, factors=[]):
        """Create a discrete Markov network.

        :param states: Node states specified either as a list of cardinalities
            (nodes become ``0..n-1``) or as a mapping ``{node: [values]}``.
        :type states: list[int] | dict
        :param edges: Optional list of undirected edges ``[(u, v), ...]``.
        :type edges: list[tuple] | None
        :param factors: Optional list of factors to attach to the model.
        :type factors: list[DiscreteFactor]
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

        :raises AssertionError: If any validation check fails.
        """
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
        """List of node identifiers in the model.

        :return: Node identifiers in a deterministic order.
        :rtype: list
        """
        return self._nodes

    @property
    def states(self):
        """Mapping from node to its allowed states.

        :return: Mapping ``{node: [values]}`` describing each node's state
            space.
        :rtype: dict
        """
        return self._states

    @states.setter
    def states(self, values):
        """Set node states from cardinalities or explicit values.

        If a list is provided, nodes are created as integers ``0..n-1`` and
        each entry indicates a node's cardinality (values ``0..card-1``). If a
        dictionary is provided, keys are node identifiers and values are the
        explicit lists of allowed states.

        :param values: Either a list of cardinalities or a mapping
            ``{node: [values]}``.
        :type values: list[int] | dict
        :raises TypeError: If ``values`` is neither a list nor a dictionary.
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

        :param node: Node identifier.
        :return: List of allowed values for ``node``.
        :rtype: list
        :raises KeyError: If ``node`` is not present.
        """
        return self._states[node]

    def card(self, node):
        """Return the cardinality of a node.

        :param node: Node identifier.
        :return: Number of allowed states for ``node``.
        :rtype: int
        :raises KeyError: If ``node`` is not present.
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

        :return: List of edges ``[(u, v), ...]``.
        :rtype: list[tuple]
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

        :param edges: Iterable of node pairs ``(u, v)``.
        :type edges: Iterable[tuple]
        """
        self._edges = list(edges)

    #
    # Factors
    #

    @property
    def factors(self):
        """List of factor potentials attached to the model.

        :return: List of :class:`DiscreteFactor`.
        :rtype: list[DiscreteFactor]
        """
        return self._factors

    @factors.setter
    def factors(self, factor_list):
        """Attach a list of factors, normalizing list-valued weights.

        Each factor is converted to the dictionary form via
        :meth:`DiscreteFactor.normalize` using the current model states.

        :param factor_list: Iterable of :class:`DiscreteFactor` objects.
        :type factor_list: Iterable[DiscreteFactor]
        """
        self._factors = [factor.normalize(self) for factor in factor_list]

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        """Create a Pyomo model for a MAP query.

        Builds an optimization model that selects one state per node to
        maximize the joint score implied by the factors, optionally under
        evidence or for a subset of variables.

        :param variables: Optional subset of variables to include.
        :type variables: list | None
        :param evidence: Optional fixed assignments ``{node: value}``.
        :type evidence: dict | None
        :param bool timing: If ``True``, collect timing diagnostics.
        :param options: Additional keyword options forwarded to the inference
            routine.
        :return: A Pyomo model representing the MAP problem.
        :rtype: object
        :raises Exception: Exceptions raised by the underlying inference
            backend are propagated.
        """
        return create_MN_map_query_model(
            pgm=self,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )


class ConstrainedDiscreteMarkovNetwork:
    """Markov network wrapper that supports custom constraints.

    Wraps a :class:`DiscreteMarkovNetwork` and an optional constraint functor
    that decorates a Pyomo model with additional feasibility constraints.
    """

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained wrapper.

        :param pgm: Base discrete Markov network to wrap.
        :type pgm: DiscreteMarkovNetwork
        :param constraints: Optional functor ``f(model) -> model`` that adds
            constraints to the Pyomo model.
        :type constraints: callable | None
        """
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        """Validate the wrapped model.

        :raises AssertionError: If the underlying model is invalid.
        """
        self.pgm.check_model()

    @property
    def nodes(self):
        """Return the underlying model's nodes.

        :return: List of node identifiers.
        :rtype: list
        """
        return self.pgm.nodes

    @property
    def constraints(self):
        """Get the current constraint functor.

        :return: The constraint functor or ``None`` if not set.
        :rtype: callable | None
        """
        return self.constraint_functor

    @constraints.setter
    def constraints(self, constraint_functor):
        """Set the constraint functor.

        :param constraint_functor: Callable that accepts a Pyomo model and
            returns it with constraints attached.
        :type constraint_functor: callable | None
        """
        self.constraint_functor = constraint_functor

    def create_constraints(self, model):
        """Apply the constraint functor to a Pyomo model, if present.

        :param model: A Pyomo model constructed from the wrapped network.
        :return: The same model, possibly with constraints added.
        :rtype: object
        """
        if self.constraint_functor is not None:
            model = self.constraint_functor(model)
        return model

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        """Create a constrained MAP Pyomo model.

        Builds the base MAP model from the wrapped network and applies the
        constraint functor before returning the final model.

        :param variables: Optional subset of variables to include.
        :type variables: list | None
        :param evidence: Optional fixed assignments ``{node: value}``.
        :type evidence: dict | None
        :param bool timing: If ``True``, collect timing diagnostics.
        :param options: Additional keyword options forwarded to the inference
            routine.
        :return: A constrained Pyomo model representing the MAP problem.
        :rtype: object
        :raises Exception: Exceptions raised by the underlying inference
            backend are propagated.
        """
        model = create_MN_map_query_model(
            pgm=self.pgm,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )
        return self.create_constraints(model)
