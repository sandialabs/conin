from conin.exceptions import InvalidInputError
from conin.dynamic_bayesian_network.expr import ExpressionVariable


class DynamicDiscreteBayesianNetwork:
    """Dynamic Bayesian network with static and time-indexed state spaces.

    Stores ordinary nodes, dynamic nodes, their corresponding state spaces,
    and conditional probability distributions that may reference time-shifted
    variables.
    """

    def __init__(self, *, cpds, states=None, dynamic_states=None, t=None):
        """Initialize a dynamic discrete Bayesian network.

        Parameters
        ----------
        cpds : list
            Non-empty list of conditional probability distributions associated
            with the model.
        states : list or dict, optional
            Mapping from static node identifiers to their allowed states, or
            a list of cardinalities for anonymous static nodes.
        dynamic_states : dict, optional
            Mapping from dynamic node identifiers to their allowed states across
            time steps. At least one of ``states`` and ``dynamic_states`` must
            be non-empty.
        t : ExpressionVariable, optional
            Symbolic time-index placeholder used in time-shifted CPD templates
            (``t``, ``t - 1``). CPDs that refer to the time index must be built
            with the same ``ExpressionVariable`` instance that is passed here;
            it is exposed afterwards as ``dbn.t``. Defaults to a fresh
            ``ExpressionVariable()``, which is only appropriate when no CPD
            refers to the time index.

        Raises
        ------
        InvalidInputError
            If ``cpds`` is empty, or if both ``states`` and ``dynamic_states``
            are empty.
        """
        cpds = list(cpds)
        if not cpds:
            raise InvalidInputError("cpds must be a non-empty list")
        if states is None:
            states = {}
        if dynamic_states is None:
            dynamic_states = {}
        if not states and not dynamic_states:
            raise InvalidInputError(
                "At least one of states and dynamic_states must be non-empty"
            )

        self._edges = []
        self._dynamic_edges = []

        if type(states) is list:
            self._nodes = list(range(len(states)))
            self._states = {i: list(range(v)) for i, v in enumerate(states)}
        elif type(states) is dict:
            self._nodes = sorted(states.keys())
            self._states = dict(states)
        else:
            raise TypeError(f"Unexpected type for states: {type(states)}")

        if type(dynamic_states) is dict:
            self._dynamic_nodes = sorted(dynamic_states.keys())
            self._dynamic_states = dict(dynamic_states)
        else:
            raise TypeError(
                f"Unexpected type for dynamic states: {type(dynamic_states)}"
            )

        self._cpds = [cpd.normalize(self) for cpd in cpds]

        self.t = t if t is not None else ExpressionVariable()

    def check_model(self):
        """Return ``True`` for the current dynamic model placeholder check.

        Full structural validation is not implemented for dynamic Bayesian
        networks yet. Static Bayesian networks produced by unrolling a dynamic
        model are validated before inference.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    #
    # Nodes
    #

    @property
    def nodes(self):
        """Return the static nodes in the model.

        Returns
        -------
        list
            Static node identifiers.
        """
        return self._nodes

    @property
    def dynamic_nodes(self):
        """Return the dynamic nodes in the model.

        Returns
        -------
        list
            Dynamic node identifiers.
        """
        return self._dynamic_nodes

    #
    # Edges
    #

    @property
    def edges(self):
        """Return the static edges in the model.

        Returns
        -------
        list
            Static edge definitions.
        """
        return self._edges

    @property
    def dynamic_edges(self):
        """Return the dynamic edges in the model.

        Returns
        -------
        list
            Dynamic edge definitions.
        """
        return self._dynamic_edges

    #
    # States
    #

    @property
    def states(self):
        """Return the state space for static nodes.

        Returns
        -------
        dict
            Mapping from static node identifiers to ordered state lists.

        Examples
        --------
        >>> from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
        >>> from conin.bayesian_network import DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states=[4, 3], cpds=[DiscreteCPD(node=0, values=[0.25] * 4)]
        ... )
        >>> ddbn.nodes
        [0, 1]
        >>> ddbn.states
        {0: [0, 1, 2, 3], 1: [0, 1, 2]}

        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states={"A": ["T", "F"], "B": [-1, 1]},
        ...     cpds=[DiscreteCPD(node="A", values=[0.5, 0.5])],
        ... )
        >>> ddbn.nodes
        ['A', 'B']
        >>> ddbn.states
        {'A': ['T', 'F'], 'B': [-1, 1]}
        """
        return dict(self._states)

    @property
    def dynamic_states(self):
        """Return the state space for dynamic nodes.

        Returns
        -------
        dict
            Mapping from dynamic node identifiers to ordered state lists.

        Examples
        --------
        >>> from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
        >>> from conin.bayesian_network import DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     dynamic_states={"A": ["T", "F"], "B": [-1, 1]},
        ...     cpds=[DiscreteCPD(node=("A", 0), values=[0.5, 0.5])],
        ... )
        >>> ddbn.dynamic_nodes
        ['A', 'B']
        >>> ddbn.dynamic_states
        {'A': ['T', 'F'], 'B': [-1, 1]}
        """
        return dict(self._dynamic_states)

    def states_of(self, node):
        """Return the states associated with a node.

        Static nodes, dynamic nodes, and time-indexed dynamic nodes represented
        as tuples are all supported.

        Parameters
        ----------
        node : Hashable or tuple
            Node identifier or time-indexed dynamic node.

        Returns
        -------
        list
            Ordered list of states for ``node``.

        Raises
        ------
        ValueError
            If ``node`` does not correspond to a known static or dynamic node.
        """
        if node in self._states:
            return self._states[node]
        elif node in self._dynamic_states:
            return self._dynamic_states[node]
        elif type(node) is tuple and node[0] in self._dynamic_states:
            return self._dynamic_states[node[0]]
        raise ValueError(f"Unexpected node value: {node}")

    def card(self, node):
        """Return the cardinality of a node.

        Parameters
        ----------
        node : Hashable or tuple
            Node identifier or time-indexed dynamic node.

        Returns
        -------
        int
            Number of allowed states for ``node``.

        Raises
        ------
        ValueError
            If ``node`` does not correspond to a known static or dynamic node.
        """
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
        """Return the conditional probability distributions.

        Returns
        -------
        list
            Conditional probability distributions stored on the model.

        Examples
        --------
        >>> from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
        >>> from conin.bayesian_network import DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states={"X": ["T", "F"], "Y": [-1, 1]},
        ...     dynamic_states={"A": ["t", "f"], "B": [2, 3]},
        ...     cpds=[
        ...         DiscreteCPD(
        ...             node=("A", None),
        ...             parents=["X"],
        ...             values={"T": {"t": 0.3, "f": 0.7}, "F": {"t": 0.4, "f": 0.6}},
        ...         ),
        ...         DiscreteCPD(
        ...             node=("B", 0),
        ...             parents=["Y"],
        ...             values={-1: {2: 0.3, 3: 0.7}, 1: {2: 0.4, 3: 0.6}},
        ...         ),
        ...     ],
        ... )
        """
        return list(self._cpds)

    def with_cpd(self, cpd):
        """Return a new network with ``cpd`` added to its collection of CPDs.

        Parameters
        ----------
        cpd : DiscreteCPD
            CPD to add to the network.

        Returns
        -------
        DynamicDiscreteBayesianNetwork
            A new network containing the existing CPDs plus ``cpd``.
        """
        return DynamicDiscreteBayesianNetwork(
            states=self.states,
            dynamic_states=self.dynamic_states,
            cpds=self.cpds + [cpd],
            t=self.t,
        )


class ConstrainedDynamicDiscreteBayesianNetwork:
    """Wrap a dynamic Bayesian network with optional constraint functors."""

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained dynamic Bayesian network.

        Parameters
        ----------
        pgm : DynamicDiscreteBayesianNetwork
            Underlying dynamic Bayesian network to be constrained.
        constraints : list[ConstraintFunctor], optional
            Constraint functors that apply additional constraints during
            inference.
        """
        self.pgm = pgm
        if constraints:
            self._constraints = constraints
        else:
            self._constraints = []

    def check_model(self):
        """Validate the underlying dynamic Bayesian network."""
        self.pgm.check_model()

    def nodes(self):
        """Return the nodes of the wrapped dynamic Bayesian network.

        Returns
        -------
        list
            Nodes maintained by the underlying dynamic Bayesian network.
        """
        return self.pgm.nodes

    @property
    def constraints(self):
        """Return the configured constraint functors.

        Returns
        -------
        list
            List of ``ConstraintFunctor`` instances that add constraints to a
            query model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set the functions that define model constraints.

        Parameters
        ----------
        constraint_list : list of ConstraintFunctor
            List of ``ConstraintFunctor`` instances that generate model
            constraints.
        """
        assert type(constraint_list) is list
        self._constraints = constraint_list
