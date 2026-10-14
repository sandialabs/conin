from conin.graphical_models import DynamicDiscreteBayesianNetwork

from .base import ConstrainedGraphicalModel


class ConstrainedDynamicDiscreteBayesianNetwork(ConstrainedGraphicalModel):
    """Dynamic Bayesian network paired with constraints.

    Parameters
    ----------
    model : DynamicDiscreteBayesianNetwork
        The model to constrain.
    constraints : iterable of Constraint, optional
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
