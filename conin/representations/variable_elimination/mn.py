"""The ``"variable_elimination"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation

from .common import VariableEliminationRepresentation


class VariableEliminationMNRepresentation(VariableEliminationRepresentation):
    """Factor model of a constrained Markov network.

    Each constraint adds an indicator factor over its scope (``0`` on
    infeasible assignments).  Constraint functions see a copy of the model's
    ``data`` (``None`` if it has none).
    """


@register_representation(
    "variable_elimination",
    DiscreteMarkovNetwork,
    families=(None, "factor_oracle", "algebraic"),
)
def _build(cmodel):
    return VariableEliminationMNRepresentation(
        pgm=cmodel.model, constraints=cmodel.constraints, data=cmodel.data
    )
