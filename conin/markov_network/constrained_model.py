class ConstrainedDiscreteMarkovNetwork:
    """Markov network that supports custom constraints.

    Wraps a ``DiscreteMarkovNetwork`` and optional ``Constraint``
    instances that apply additional feasibility constraints.
    """

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained wrapper.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork
            Underlying probabilistic graphical model to constrain.
        constraints : list[Constraint] or None, optional
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
            List of ``Constraint`` instances that add constraints to an
            inference model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set the functions used to define model constraints.

        Parameters
        ----------
        constraint_list : list of Constraint
            List of ``Constraint`` instances that generate model
            constraints.
        """
        assert type(constraint_list) is list
        self._constraints = constraint_list
