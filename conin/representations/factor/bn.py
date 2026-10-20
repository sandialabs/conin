"""The ``"factor"`` representation of a (constrained) Bayesian network."""

from conin.graphical_models import DiscreteBayesianNetwork
from conin.representations.base import register_representation

from .common import FactorRepresentation


class FactorBNRepresentation(FactorRepresentation):
    """Factor model of a constrained Bayesian network.

    Each constraint adds the CPD of a binary node ``(name, -1)`` whose parents
    are its scope; :attr:`constraint_evidence` fixes those nodes to ``1``.
    Constraint functions see ``data = None``.
    """


@register_representation(
    "factor", DiscreteBayesianNetwork, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel):
    return FactorBNRepresentation(pgm=cmodel.model, constraints=cmodel.constraints)
