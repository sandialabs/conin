from conin.dynamic_bayesian_network.expr import ExpressionVariable


class DynamicDiscreteBayesianNetwork:
    """Dynamic Bayesian network with static and time-indexed state spaces.

    Stores ordinary nodes, dynamic nodes, their corresponding state spaces,
    and conditional probability distributions that may reference time-shifted
    variables.
    """

    def __init__(self, *, states={}, dynamic_states={}, cpds=[]):
        """Initialize a dynamic discrete Bayesian network.

        Parameters
        ----------
        states : dict, optional
            Mapping from static node identifiers to their allowed states.
        dynamic_states : dict, optional
            Mapping from dynamic node identifiers to their allowed states across
            time steps.
        cpds : list, optional
            Conditional probability distributions associated with the model.
        """
        self._nodes = []
        self._dynamic_nodes = []
        self._edges = []
        self._dynamic_edges = []
        self._states = states
        self._dynamic_states = dynamic_states
        self._cpds = cpds
        self.t = ExpressionVariable()

    def check_model(self):
        """Validate the dynamic model structure.

        Returns
        -------
        bool
            Always returns ``True``.
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
        """
        return self._states

    @states.setter
    def states(self, values):
        """Set the state space for static nodes.

        If a list is provided, nodes are created as integers ``0..n-1`` and
        each entry gives that node's cardinality with state values
        ``0..card-1``. If a dictionary is provided, keys are node identifiers
        and values are explicit state lists.

        Parameters
        ----------
        values : list or dict
            Cardinalities or explicit state definitions for static nodes.

        Raises
        ------
        TypeError
            If ``values`` is neither a list nor a dictionary.

        Examples
        --------
        >>> ddbn = DynamicDiscreteBayesianNetwork()
        >>> ddbn.states = [4, 3]
        >>> ddbn.nodes
        [0, 1]
        >>> ddbn.states
        {0: [0, 1, 2, 3], 1: [0, 1, 2]}

        >>> ddbn = DynamicDiscreteBayesianNetwork()
        >>> ddbn.states = {"A": ["T", "F"], "B": [-1, 1]}
        >>> ddbn.nodes
        ['A', 'B']
        >>> ddbn.states
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

    @property
    def dynamic_states(self):
        """Return the state space for dynamic nodes.

        Returns
        -------
        dict
            Mapping from dynamic node identifiers to ordered state lists.
        """
        return self._dynamic_states

    @dynamic_states.setter
    def dynamic_states(self, values):
        """Set the state space for dynamic nodes.

        Parameters
        ----------
        values : dict
            Mapping from dynamic node identifiers to explicit state lists.

        Raises
        ------
        TypeError
            If ``values`` is not a dictionary.

        Examples
        --------
        >>> ddbn = DynamicDiscreteBayesianNetwork()
        >>> ddbn.dynamic_states = {"A": ["T", "F"], "B": [-1, 1]}
        >>> ddbn.dynamic_nodes
        ['A', 'B']
        >>> ddbn.dynamic_states
        {'A': ['T', 'F'], 'B': [-1, 1]}
        """
        if type(values) is dict:
            self._dynamic_nodes = sorted(values.keys())
            self._dynamic_states = values

        else:
            raise TypeError(f"Unexpected type for dynamic states: {type(values)}")

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
        """
        return self._cpds

    @cpds.setter
    def cpds(self, cpd_list):
        """Store CPDs after normalizing their internal representation.

        Parameters
        ----------
        cpd_list : list
            Iterable of CPDs to attach to the dynamic network.

        Examples
        --------
        >>> ddbn = DynamicDiscreteBayesianNetwork()
        >>> ddbn.states = {"X": ["T", "F"], "Y": [-1, 1]}
        >>> ddbn.dynamic_states = {"A": ["t", "f"], "B": [2, 3]}
        >>> c1 = DiscreteCPD(
        ...     node=("A", None),
        ...     parents=["X"],
        ...     values={"T": {"t": 0.3, "f": 0.7}, "F": {"t": 0.4, "f": 0.6}},
        ... )
        >>> c2 = DiscreteCPD(
        ...     node=("B", 0),
        ...     parents=["Y"],
        ...     values={-1: {2: 0.3, 3: 0.7}, 1: {2: 0.4, 3: 0.6}},
        ... )
        >>> ddbn.cpds = [c1, c2]
        """
        self._cpds = [cpd.normalize(self) for cpd in cpd_list]


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
        return self.pgm.nodes()

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
