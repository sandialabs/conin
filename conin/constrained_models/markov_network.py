from conin.graphical_models import DiscreteMarkovNetwork

from .base import ConstrainedGraphicalModel


class ConstrainedDiscreteMarkovNetwork(ConstrainedGraphicalModel):
    """Markov network paired with constraints.

    Parameters
    ----------
    model : DiscreteMarkovNetwork
        The model to constrain.
    constraints : Constraint or iterable of Constraint, optional
        Constraints of one family, applied during inference.
    data : object, optional
        Application data passed to the representation builders.
    """

    __slots__ = ()

    model_type = DiscreteMarkovNetwork

    def check_model(self):
        """Validate the underlying model."""
        self.model.check_model()

    @property
    def nodes(self):
        """Return the underlying model's nodes."""
        return self.model.nodes

    def states_of(self, node):
        """Return the allowed states for ``node`` in the underlying model."""
        return self.model.states_of(node)

    def is_feasible(self, assignment):
        """Return whether a total assignment satisfies the constraints.

        Parameters
        ----------
        assignment : dict
            A state for every node of the model.

        Returns
        -------
        bool
            ``True`` if every constraint is satisfied (always, without
            constraints).  ``"oracle"`` and ``"factor_oracle"`` predicates see
            the assignment (restricted to their node scope for factor oracles)
            and ``data=None``; ``"algebraic"`` constraints are compiled into a
            predicate (this needs smoek).

        Raises
        ------
        InvalidInputError
            If ``assignment`` is not a total assignment of the model.
        NotImplementedError
            For ``"pyomo"`` and ``"toulbar2"`` constraints, whose functions can
            add variables to the backend model.
        """
        from .feasibility import StateSpace, check_constraints, check_total_assignment

        states = self.model.states
        check_total_assignment(states, assignment)
        return check_constraints(
            self,
            assignment=assignment,
            factor_assignment=assignment,
            state_space=StateSpace(states),
            data=None,
        )
