"""The ``"pgmpy"`` representation of a (constrained) dynamic Bayesian network."""

from conin.graphical_models import DynamicDiscreteBayesianNetwork, create_bn_from_dbn
from conin.representations.base import register_representation
from conin.representations.unrolled import UnrolledDBNMixin, dbn_time_range

from .common import PgmpyRepresentation


class PgmpyDBNRepresentation(UnrolledDBNMixin, PgmpyRepresentation):
    """Factor model of a constrained DBN unrolled over ``start..stop``.

    Structure params: ``start`` and ``stop``.  ``pgm`` is the unrolled BN;
    constraints are added as for a Bayesian network.  Constraint functions see
    ``data = Munch(start, stop, T)`` with ``T = list(range(start, stop + 1))``.
    """


def build_pgmpy_dbn_representation(
    dbn, constraints=(), *, start, stop, cls=PgmpyDBNRepresentation, **kwds
):
    """Unroll ``dbn`` over ``start..stop`` and build its ``"pgmpy"`` representation.

    ``kwds`` are passed to ``cls`` (``data``, and ``hidden_to_internal`` for
    the HMM subclass).
    """
    bn = create_bn_from_dbn(dbn=dbn, start=start, stop=stop)
    return cls(pgm=bn, constraints=constraints, start=start, stop=stop, **kwds)


@register_representation(
    "pgmpy",
    DynamicDiscreteBayesianNetwork,
    families=(None, "factor_oracle", "algebraic"),
)
def _build(cmodel, *, start=0, stop=None):
    start, stop = dbn_time_range(start, stop)
    return build_pgmpy_dbn_representation(
        cmodel.model, cmodel.constraints, start=start, stop=stop, data=cmodel.data
    )
