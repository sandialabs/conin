"""The ``"variable_elimination"`` representation of a (constrained) Bayesian network."""

from conin.graphical_models import DiscreteBayesianNetwork
from conin.representations.base import register_representation

from .common import VariableEliminationRepresentation


class VariableEliminationBNRepresentation(VariableEliminationRepresentation):
    """Factor model of a constrained Bayesian network.

    :attr:`ve_model` is the Markov network of the CPDs, and each constraint
    adds an indicator factor over its scope (``0`` on infeasible
    assignments).  Constraint functions see a copy of the model's ``data``
    (``None`` if it has none).
    """


@register_representation(
    "variable_elimination",
    DiscreteBayesianNetwork,
    families=(None, "factor_oracle", "smoek"),
)
def _build(cmodel):
    return VariableEliminationBNRepresentation(
        pgm=cmodel.model, constraints=cmodel.constraints, data=cmodel.data
    )
