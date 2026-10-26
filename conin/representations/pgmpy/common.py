"""Shared code of the ``"pgmpy"`` representations (variable elimination).

- :func:`apply_pgmpy_constraints`: the single constraint applier for the
  ``"pgmpy"`` backend families (``"factor_oracle"`` and ``"algebraic"``, the latter
  via :func:`~conin.constraints.algebraic.create_oracle_constraints_from_algebraic`).
  It materialises each constraint into a factor (Markov network) or a CPD of
  a new binary node (Bayesian network) with
  :func:`~conin.representations.pgmpy.materialise_factor_oracle`;
- :class:`PgmpyRepresentation`: the base class of the MN/BN/DBN/HMM
  ``"pgmpy"`` representations.
"""

import copy
import dataclasses

import munch

from conin.pgmpy import convert_conin_to_pgmpy_bn, convert_conin_to_pgmpy_mn
from conin.constraints.algebraic import create_oracle_constraints_from_algebraic
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from conin.io import save_model
from conin.representations.base import Representation, apply_constraints
from conin.util.constraint_data import merge_constraint_data

from .materialise import materialise_factor_oracle


def apply_pgmpy_constraints(*, pgm, constraints, data=None):
    """Add constraints of a family of the ``"pgmpy"`` backend to ``pgm``.

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
        For constraints of several families or of another family, or a model
        that is neither a ``DiscreteMarkovNetwork`` nor a
        ``DiscreteBayesianNetwork``.
    ImportError
        For algebraic constraints without smoek.
    """
    result = apply_constraints(
        constraints,
        {
            "factor_oracle": lambda cons: _materialise_all(pgm, cons, data),
            "algebraic": lambda cons: _materialise_all(
                pgm,
                create_oracle_constraints_from_algebraic(
                    pgm=pgm, constraints=cons, data=data
                ),
                data,
            ),
        },
        backend="pgmpy",
    )
    return (pgm, {}) if result is None else result


def _materialise_all(pgm, constraints, data):
    """Add each factor-oracle constraint to ``pgm``; return ``(pgm, evidence)``."""
    if isinstance(pgm, DiscreteBayesianNetwork):
        is_bn = True
    elif isinstance(pgm, DiscreteMarkovNetwork):
        is_bn = False
    else:
        raise TypeError(f"Unexpected model type {type(pgm)} for VariableElimination")

    evidence = {}
    for con in constraints:
        if is_bn:
            cpd = materialise_factor_oracle(con, pgm, data)
            cpd = dataclasses.replace(cpd, node=(cpd.node, -1))
            pgm = pgm.with_cpd(cpd)
            evidence[cpd.node] = 1
        else:
            # The factor is 0 on infeasible assignments, so no evidence is
            # needed.
            pgm = pgm.with_factor(materialise_factor_oracle(con, pgm, data))
    return pgm, evidence


class PgmpyRepresentation(Representation):
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
    data : object or None
        The constrained model's application data.  The constraint functions
        see a copy of it with the fields of :meth:`constraint_data` set.
    """

    kind = "pgmpy"

    def __init__(self, *, pgm, constraints, data=None):
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(pgm.nodes)
        self.data = data
        self.ve_model, self.constraint_evidence = apply_pgmpy_constraints(
            pgm=pgm,
            constraints=self.constraints,
            data=merge_constraint_data(data, self.constraint_data()),
        )

    def structure_fields(self):
        """The constraint ``data`` fields of the structure params (none here).

        The DBN and HMM subclasses return ``start``, ``stop`` and ``T`` (or
        ``hmm``); see :mod:`conin.representations.unrolled`.
        """
        return munch.Munch()

    def constraint_data(self):
        """The fields this representation adds to the constraint ``data``.

        :meth:`structure_fields`, or ``None`` if there are none (MN/BN).  The
        constraint functions see these fields set on a copy of the constrained
        model's ``data`` (a plain copy for MN/BN; ``None`` if the model has no
        ``data`` and there are no fields).  The constraints are materialised
        once, when the representation is built, so the query evidence is not
        passed.
        """
        return self.structure_fields() or None

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
