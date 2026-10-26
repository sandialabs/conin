"""The ``"pgmpy"`` representation of a (constrained) Bayesian network."""

from conin.graphical_models import DiscreteBayesianNetwork
from conin.representations.base import register_representation

from .common import PgmpyRepresentation


class PgmpyBNRepresentation(PgmpyRepresentation):
    """Factor model of a constrained Bayesian network.

    Each constraint adds the CPD of a binary node ``(name, -1)`` whose parents
    are its scope; :attr:`constraint_evidence` fixes those nodes to ``1``.
    Constraint functions see a copy of the model's ``data`` (``None`` if it
    has none).
    """


@register_representation(
    "pgmpy", DiscreteBayesianNetwork, families=(None, "factor_oracle", "algebraic")
)
def _build(cmodel):
    return PgmpyBNRepresentation(
        pgm=cmodel.model, constraints=cmodel.constraints, data=cmodel.data
    )
