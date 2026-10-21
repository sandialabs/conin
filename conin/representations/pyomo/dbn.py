"""The ``"pyomo"`` representation of a (constrained) dynamic Bayesian network."""

import munch

from conin.graphical_models import DynamicDiscreteBayesianNetwork, create_bn_from_dbn
from conin.representations.base import register_representation

from .bn import build_pyomo_bn_representation
from .common import PyomoFactorRepresentation


class PyomoDBNRepresentation(PyomoFactorRepresentation):
    """Factor-repn IP of a constrained DBN unrolled over ``start..stop``.

    Structure params: ``start`` and ``stop``.  Constraint
    functions see ``data = Munch(start, stop, variables, evidence, T)`` with
    ``T = list(range(start, stop + 1))``.  ``pgm`` is the unrolled BN.
    """

    def __init__(self, *, start, stop, **kwds):
        super().__init__(**kwds)
        self.start = start
        self.stop = stop

    def constraint_data(self, *, evidence=None, variables=None):
        return munch.Munch(
            start=self.start,
            stop=self.stop,
            variables=variables,
            evidence=evidence,
            T=list(range(self.start, self.stop + 1)),
        )


def build_pyomo_dbn_representation(dbn, constraints=(), *, start, stop, data=None):
    """Unroll ``dbn`` over ``start..stop`` and build its ``"pyomo"`` representation."""
    bn = create_bn_from_dbn(dbn=dbn, start=start, stop=stop)
    base = build_pyomo_bn_representation(bn)
    return PyomoDBNRepresentation(
        S=base.S,
        J=base.J,
        v=base.v,
        w=base.w,
        var_index_map=base.var_index_map,
        pgm=bn,
        constraints=constraints,
        nodes=base.nodes,
        start=start,
        stop=stop,
        data=data,
    )


@register_representation(
    "pyomo", DynamicDiscreteBayesianNetwork, families=(None, "pyomo", "algebraic")
)
def _build(cmodel, *, start=0, stop=1):
    return build_pyomo_dbn_representation(
        cmodel.model, cmodel.constraints, start=start, stop=stop, data=cmodel.data
    )
