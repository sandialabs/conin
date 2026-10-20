"""The ``"factor"`` representation of a (constrained) dynamic Bayesian network."""

import munch

from conin.graphical_models import DynamicDiscreteBayesianNetwork, create_bn_from_dbn
from conin.representations.base import register_representation

from .common import FactorRepresentation


class FactorDBNRepresentation(FactorRepresentation):
    """Factor model of a constrained DBN unrolled over ``start..stop``.

    Structure params: ``start`` and ``stop``.  ``pgm`` is the unrolled BN;
    constraints are added as for a Bayesian network.  Constraint functions see
    ``data = Munch(T)`` with ``T = list(range(start, stop + 1))``.
    """

    def __init__(self, *, start, stop, **kwds):
        self.start = start
        self.stop = stop
        super().__init__(**kwds)

    def constraint_data(self):
        return munch.Munch(T=list(range(self.start, self.stop + 1)))


def build_factor_dbn_representation(
    dbn, constraints=(), *, start, stop, cls=FactorDBNRepresentation
):
    """Unroll ``dbn`` over ``start..stop`` and build its ``"factor"`` representation."""
    bn = create_bn_from_dbn(dbn=dbn, start=start, stop=stop)
    return cls(pgm=bn, constraints=constraints, start=start, stop=stop)


@register_representation(
    "factor",
    DynamicDiscreteBayesianNetwork,
    families=(None, "factor_oracle", "algebraic"),
)
def _build(cmodel, *, start=0, stop=1):
    return build_factor_dbn_representation(
        cmodel.model, cmodel.constraints, start=start, stop=stop
    )
