"""The ``"sparow"`` representation of a (constrained) Bayesian network."""

from conin.graphical_models import DiscreteBayesianNetwork
from conin.representations.base import register_representation
import conin.representations.pyomo

from .common import SparowFactorRepresentation
from .mn import build_sparow_mn_representation


class SparowBNRepresentation(SparowFactorRepresentation):
    """Factor-repn IP of a constrained Bayesian network.

    The IP is built on the BN's Markov-network skeleton (one factor per CPD).
    Constraint functions see ``data = Munch(variables=..., evidence=...)``.
    """


def build_sparow_bn_representation(
    bn, constraints=(), *, cls=SparowBNRepresentation, **kwds
):
    """Build a factor-repn IP representation of a :class:`DiscreteBayesianNetwork`.

    It reuses the ``"sparow"`` representation of the skeleton Markov network.
    ``kwds`` are passed to ``cls`` (``data``, and the structure fields of the
    DBN/HMM subclasses); ``pgm`` defaults to ``bn``.
    """
    base = build_sparow_mn_representation(conin.representation.pyomo.factor_repn.skeleton_mn_of_bn(bn))
    return cls(
        S=base.S,
        J=base.J,
        v=base.v,
        w=base.w,
        var_index_map=base.var_index_map,
        pgm=kwds.pop("pgm", bn),
        constraints=constraints,
        nodes=bn.nodes,
        **kwds,
    )


@register_representation(
    "sparow", DiscreteBayesianNetwork, families=(None, "pyomo", "smoek")
)
def _build(cmodel):
    return build_sparow_bn_representation(
        cmodel.model, cmodel.constraints, data=cmodel.data
    )
