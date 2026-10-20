"""Shared code of the ``"factor"`` representations (variable elimination).

- :func:`apply_factor_constraints`: the single constraint applier for the
  factor-backend families (``"factor_oracle"`` and ``"algebraic"``, the latter
  via :func:`~conin.constraints.algebraic.create_oracle_constraints_from_algebraic`).
  It materialises each constraint into a factor (Markov network) or a CPD of
  a new binary node (Bayesian network) with
  :func:`~conin.representations.factor.materialise_factor_oracle`;
- :class:`FactorRepresentation`: the base class of the MN/BN/DBN/HMM
  ``"factor"`` representations.
"""

import copy
import dataclasses

from conin.common.pgmpy import convert_conin_to_pgmpy_bn, convert_conin_to_pgmpy_mn
from conin.constraints import FactorOracleConstraint
from conin.constraints.algebraic import (
    AlgebraicConstraint,
    create_oracle_constraints_from_algebraic,
)
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from conin.io import save_model
from conin.representations.base import Representation

from .materialise import materialise_factor_oracle


def apply_factor_constraints(*, pgm, constraints, data=None):
    """Add constraints of a factor-backend family to ``pgm``.

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The (possibly unrolled) model the constraints refer to.
    constraints : sequence of FactorOracleConstraint or AlgebraicConstraint
        Constraints of one family.  Algebraic constraints are first combined
        into one :class:`FactorOracleConstraint`.
    data : object, optional
        Passed to the constraint functions (and to callable node scopes).

    Returns
    -------
    (GraphicalModel, dict)
        The new model and the evidence that enforces the constraints.  For a
        Markov network, each constraint becomes a factor over its scope (``0``
        on infeasible assignments) and the evidence is empty.  For a Bayesian
        network, each constraint becomes the CPD of a new binary node
        ``(constraint.name, -1)`` whose parents are the scope, and the evidence
        fixes each such node to ``1``.  ``pgm`` itself is not changed.

    Raises
    ------
    TypeError
        For a constraint that is not a :class:`FactorOracleConstraint` (after
        the algebraic translation), or a model that is neither a
        ``DiscreteMarkovNetwork`` nor a ``DiscreteBayesianNetwork``.
    """
    if len(constraints) == 0:
        return pgm, {}

    if isinstance(constraints[0], AlgebraicConstraint):
        constraints = create_oracle_constraints_from_algebraic(
            pgm=pgm, constraints=constraints, data=data
        )

    if isinstance(pgm, DiscreteBayesianNetwork):
        is_bn = True
    elif isinstance(pgm, DiscreteMarkovNetwork):
        is_bn = False
    else:
        raise TypeError(f"Unexpected model type {type(pgm)} for VariableElimination")

    evidence = {}
    for con in constraints:
        if not isinstance(con, FactorOracleConstraint):
            raise TypeError(
                f"Unexpected constraint type {type(con)} for VariableElimination"
            )
        if is_bn:
            cpd = materialise_factor_oracle(con, pgm, data)
            cpd = dataclasses.replace(cpd, node=(cpd.node, -1))
            pgm = pgm.with_cpd(cpd)
            evidence[cpd.node] = 1
        else:
            # The factor is 0 on infeasible assignments, so no evidence is
            # needed.  (Before PR 4e the last scope node was fixed to 1.)
            pgm = pgm.with_factor(materialise_factor_oracle(con, pgm, data))
    return pgm, evidence


class FactorRepresentation(Representation):
    """The factor model of a constrained MN/BN/DBN/HMM, used by variable elimination.

    The constraints are materialised once, when the representation is built
    (their ``data`` holds no evidence): :attr:`ve_model` is :attr:`pgm` plus
    one factor (MN) or one CPD of a new binary node (BN) per constraint, and
    :attr:`constraint_evidence` fixes those CPD nodes to ``1``.

    Evidence and the other query options are not stored:
    :meth:`query_evidence` returns a new dict per query and
    :meth:`create_model` a fresh pgmpy model, so one representation can serve
    several queries.

    Attributes
    ----------
    pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The model the constraint applier sees: the MN, the BN, or the
        unrolled BN of a DBN/HMM.
    constraints : tuple
        The user constraints (family ``"factor_oracle"`` or ``"algebraic"``).
    ve_model : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        ``pgm`` with the materialised constraints; what variable elimination
        runs on and what ``write_uai_file`` writes.
    constraint_evidence : dict
        ``{(constraint_name, -1): 1}`` for each constraint CPD node (BN-based
        models); empty for Markov networks.
    nodes : tuple
        The nodes of ``pgm`` (without the constraint nodes).
    """

    kind = "factor"

    def __init__(self, *, pgm, constraints):
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(pgm.nodes)
        self.ve_model, self.constraint_evidence = apply_factor_constraints(
            pgm=pgm, constraints=self.constraints, data=self.constraint_data()
        )

    def constraint_data(self):
        """The ``data`` passed to the constraint functions (``None`` for MN/BN).

        Subclasses return their structure fields (``T``, ``hmm.T``).  Neither
        the query evidence nor the constrained model's own ``data`` is passed
        (as before PR 4e).
        """
        return None

    def query_evidence(self, evidence=None):
        """Return a new dict: ``evidence`` plus :attr:`constraint_evidence`."""
        query = copy.copy(evidence) if evidence else {}
        query.update(self.constraint_evidence)
        return query

    def create_model(self, *, write_uai_file=None):
        """Convert :attr:`ve_model` into a fresh pgmpy model for one query.

        Parameters
        ----------
        write_uai_file : str, optional
            Also save :attr:`ve_model` to this file with
            :func:`conin.io.save_model`.

        Returns
        -------
        pgmpy.models.DiscreteMarkovNetwork or pgmpy.models.DiscreteBayesianNetwork
        """
        if write_uai_file:
            save_model(self.ve_model, write_uai_file)
        if isinstance(self.ve_model, DiscreteBayesianNetwork):
            return convert_conin_to_pgmpy_bn(self.ve_model)
        return convert_conin_to_pgmpy_mn(self.ve_model)
