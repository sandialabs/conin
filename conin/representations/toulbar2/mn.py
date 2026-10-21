"""The ``"toulbar2"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation

from .common import Toulbar2FactorRepresentation


class Toulbar2MNRepresentation(Toulbar2FactorRepresentation):
    """toulbar2 CFN of a constrained Markov network (read from a MARKOV UAI file).

    Constraint functions see ``data = Munch(variables=..., evidence=...)``.
    """


@register_representation(
    "toulbar2", DiscreteMarkovNetwork, families=(None, "toulbar2", "algebraic")
)
def _build(cmodel):
    mn = cmodel.model
    return Toulbar2MNRepresentation(
        uai_model=mn, pgm=mn, constraints=cmodel.constraints, data=cmodel.data
    )
