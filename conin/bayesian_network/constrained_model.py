class ConstrainedDiscreteBayesianNetwork:
    """Wrap a Bayesian network with optional constraint functors."""

    def __init__(self, pgm, constraints=None):
        """Initialize the constrained Bayesian network.

        Parameters
        ----------
        pgm : DiscreteBayesianNetwork
            Underlying Bayesian network to be constrained.
        constraints : list[Constraint], optional
            Constraint functors that apply additional constraints during
            inference.
        """
        self.pgm = pgm
        if constraints:
            self._constraints = constraints
        else:
            self._constraints = []

    def check_model(self):
        """Validate the underlying Bayesian network."""
        self.pgm.check_model()

    @property
    def nodes(self):
        """Return the nodes of the wrapped Bayesian network.

        Returns
        -------
        list
            Nodes maintained by the underlying Bayesian network.
        """
        return self.pgm.nodes

    @property
    def constraints(self):
        """Return the configured constraint functors.

        Returns
        -------
        list
            List of ``Constraint`` instances that add constraints to a
            query model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set the functions that define model constraints.

        Parameters
        ----------
        constraint_list : list of Constraint
            List of ``Constraint`` instances that generate model
            constraints.
        """
        assert type(constraint_list) is list
        self._constraints = constraint_list
