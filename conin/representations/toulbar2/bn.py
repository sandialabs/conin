"""The ``"toulbar2"`` representation of a (constrained) Bayesian network."""

from conin.graphical_models import DiscreteBayesianNetwork
from conin.representations.base import register_representation

from .common import Toulbar2FactorRepresentation


class Toulbar2BNRepresentation(Toulbar2FactorRepresentation):
    """toulbar2 CFN of a constrained Bayesian network (read from a BAYES UAI file).

    Constraint functions see ``data = Munch(variables=..., evidence=...)``.
    """


def build_toulbar2_bn_representation(
    bn, constraints=(), *, cls=Toulbar2BNRepresentation, **kwds
):
    """Build a ``"toulbar2"`` representation of a :class:`DiscreteBayesianNetwork`.

    ``kwds`` are passed to ``cls`` (the DBN/HMM subclasses add structure fields).
    """
    return cls(uai_model=bn, pgm=kwds.pop("pgm", bn), constraints=constraints, **kwds)


@register_representation(
    "toulbar2", DiscreteBayesianNetwork, families=(None, "toulbar2", "algebraic")
)
def _build(cmodel):
    return build_toulbar2_bn_representation(cmodel.model, cmodel.constraints)
