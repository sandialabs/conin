"""The ``"toulbar2"`` representation of a (constrained) dynamic Bayesian network."""

from conin.graphical_models import DynamicDiscreteBayesianNetwork, create_bn_from_dbn
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledDBNMixin, dbn_time_range

from .bn import build_toulbar2_bn_representation
from .common import Toulbar2FactorRepresentation


class Toulbar2DBNRepresentation(UnrolledDBNMixin, Toulbar2FactorRepresentation):
    """toulbar2 CFN of a constrained DBN unrolled over ``start..stop``.

    Structure params: ``start`` and ``stop``.  Constraint
    functions see ``data = Munch(start, stop, variables, evidence, T)`` with
    ``T = list(range(start, stop + 1))``.  ``pgm`` and ``uai_model`` are the
    unrolled BN.
    """


def build_toulbar2_dbn_representation(
    dbn, constraints=(), *, start, stop, cls=Toulbar2DBNRepresentation, **kwds
):
    """Unroll ``dbn`` over ``start..stop`` and build its ``"toulbar2"`` representation.

    ``kwds`` are passed to ``cls`` (the HMM subclass adds ``pgm`` and
    ``hidden_to_internal``).
    """
    bn = create_bn_from_dbn(dbn=dbn, start=start, stop=stop)
    return build_toulbar2_bn_representation(
        bn, constraints, cls=cls, start=start, stop=stop, **kwds
    )


@register_representation(
    "toulbar2",
    DynamicDiscreteBayesianNetwork,
    families=(None, "toulbar2", "algebraic"),
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = dbn_time_range(start, stop)
    return build_toulbar2_dbn_representation(
        cmodel.model, cmodel.constraints, start=start, stop=stop, data=cmodel.data
    )
