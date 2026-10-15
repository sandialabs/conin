"""The ``"pyomo"`` representation of a (constrained) Markov network."""

from conin.graphical_models import DiscreteMarkovNetwork
from conin.representations.base import register_representation

from .common import PyomoFactorRepresentation, factor_representation_of_mn


class PyomoMNRepresentation(PyomoFactorRepresentation):
    """Factor-repn IP of a constrained Markov network.

    Constraint functions see ``data = Munch(variables=..., evidence=...)``.
    """


def build_pyomo_mn_representation(mn, constraints=()):
    """Build the ``"pyomo"`` representation of a :class:`DiscreteMarkovNetwork`."""
    S, J, v, w, var_index_map = factor_representation_of_mn(mn)
    return PyomoMNRepresentation(
        S=S,
        J=J,
        v=v,
        w=w,
        var_index_map=var_index_map,
        pgm=mn,
        constraints=constraints,
        nodes=mn.nodes,
    )


@register_representation(
    "pyomo", DiscreteMarkovNetwork, families=(None, "pyomo", "algebraic")
)
def _build(cmodel):
    return build_pyomo_mn_representation(cmodel.model, cmodel.constraints)
