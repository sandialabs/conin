from conin.graphical_models import DynamicDiscreteBayesianNetwork

from .base import ConstrainedGraphicalModel


class ConstrainedDynamicDiscreteBayesianNetwork(ConstrainedGraphicalModel):
    """Dynamic Bayesian network paired with constraints.

    Parameters
    ----------
    model : DynamicDiscreteBayesianNetwork
        The model to constrain.
    constraints : Constraint or iterable of Constraint, optional
        Constraints of one family, applied during inference.
    data : object, optional
        Application data passed to the representation builders.
    """

    __slots__ = ()

    model_type = DynamicDiscreteBayesianNetwork

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
        """Return whether a total assignment of the unrolled DBN satisfies the constraints.

        Parameters
        ----------
        assignment : dict
            A state for every static node and for every ``(name, t)`` with
            ``name`` a dynamic node and ``t`` in ``0..stop``.  ``stop`` is the
            largest time in the keys.

        Returns
        -------
        bool
            ``True`` if every constraint is satisfied (always, without
            constraints).  Constraint functions see
            ``data = Munch(T=list(range(stop + 1)))``, as for variable
            elimination; ``"algebraic"`` constraints need smoek.

        Raises
        ------
        InvalidInputError
            If ``assignment`` is not a total assignment over ``0..stop``.
        NotImplementedError
            For ``"pyomo"`` and ``"toulbar2"`` constraints, whose functions can
            add variables to the backend model.
        """
        from .feasibility import (
            StateSpace,
            check_constraints,
            check_total_assignment,
            dbn_constraint_data,
            dbn_horizon,
            dbn_states,
        )

        stop = dbn_horizon(self.model, assignment)
        states = dbn_states(self.model, stop)
        check_total_assignment(states, assignment)
        return check_constraints(
            self,
            assignment=assignment,
            factor_assignment=assignment,
            state_space=StateSpace(states),
            data=dbn_constraint_data(stop),
        )
