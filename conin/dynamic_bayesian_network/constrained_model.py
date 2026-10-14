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
