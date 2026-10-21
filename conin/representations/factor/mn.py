"""The ``"factor"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation

from .common import FactorRepresentation


class FactorMNRepresentation(FactorRepresentation):
    """Factor model of a constrained Markov network.

    Each constraint adds a factor over its scope (``0`` on infeasible
    assignments); :attr:`constraint_evidence` is empty.  Constraint functions
    see ``data = None``.
    """


@register_representation(
    "factor", DiscreteMarkovNetwork, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel):
    return FactorMNRepresentation(
        pgm=cmodel.model, constraints=cmodel.constraints, data=cmodel.data
    )
