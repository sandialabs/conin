"""The ``"sparow"`` representation of a (constrained) dynamic Bayesian network."""

from conin.graphical_models import DynamicDiscreteBayesianNetwork, create_bn_from_dbn
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledDBNMixin, dbn_time_range

from .bn import build_sparow_bn_representation
from .common import SparowFactorRepresentation


class SparowDBNRepresentation(UnrolledDBNMixin, SparowFactorRepresentation):
    """Factor-repn IP of a constrained DBN unrolled over ``start..stop``.

    Structure params: ``start`` and ``stop``.  Constraint
    functions see ``data = Munch(start, stop, variables, evidence, T)`` with
    ``T = list(range(start, stop + 1))``.  ``pgm`` is the unrolled BN.
    """


def build_sparow_dbn_representation(
    dbn, constraints=(), *, start, stop, cls=SparowDBNRepresentation, **kwds
):
    """Unroll ``dbn`` over ``start..stop`` and build its ``"sparow"`` representation.

    ``kwds`` are passed to ``cls`` (the HMM subclass adds ``pgm`` and
    ``hidden_to_internal``).
    """
    bn = create_bn_from_dbn(dbn=dbn, start=start, stop=stop)
    return build_sparow_bn_representation(
        bn, constraints, cls=cls, start=start, stop=stop, **kwds
    )


@register_representation(
    "sparow", DynamicDiscreteBayesianNetwork, families=(None, "pyomo", "smoek")
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = dbn_time_range(start, stop)
    return build_sparow_dbn_representation(
        cmodel.model, cmodel.constraints, start=start, stop=stop, data=cmodel.data
    )
