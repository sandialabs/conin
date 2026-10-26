"""The ``"pgmpy"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation

from .common import PgmpyRepresentation


class PgmpyMNRepresentation(PgmpyRepresentation):
    """Factor model of a constrained Markov network.

    Each constraint adds a factor over its scope (``0`` on infeasible
    assignments); :attr:`constraint_evidence` is empty.  Constraint functions
    see a copy of the model's ``data`` (``None`` if it has none).
    """


@register_representation(
    "pgmpy", DiscreteMarkovNetwork, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel):
    return PgmpyMNRepresentation(
        pgm=cmodel.model, constraints=cmodel.constraints, data=cmodel.data
    )
