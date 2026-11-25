"""The ``"sparow"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation
import conin.representations.pyomo

from .common import SparowFactorRepresentation


class SparowMNRepresentation(SparowFactorRepresentation):
    """Factor-repn IP of a constrained Markov network.

    Constraint functions see ``data = Munch(variables=..., evidence=...)``.
    """


def build_sparow_mn_representation(mn, constraints=(), *, data=None):
    """Build the ``"sparow"`` representation of a :class:`DiscreteMarkovNetwork`."""
    S, J, v, w, var_index_map = (
        conin.representations.pyomo.mn.factor_representation_of_mn(mn)
    )
    return SparowMNRepresentation(
        S=S,
        J=J,
        v=v,
        w=w,
        var_index_map=var_index_map,
        pgm=mn,
        constraints=constraints,
        nodes=mn.nodes,
        data=data,
    )


@register_representation(
    "sparow", DiscreteMarkovNetwork, families=(None, "pyomo", "smoek")
)
def _build(cmodel):
    return build_sparow_mn_representation(
        cmodel.model, cmodel.constraints, data=cmodel.data
    )
