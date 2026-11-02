"""Shared code of the ``"variable_elimination"`` representations.

- :func:`apply_variable_elimination_constraints`: the single constraint
  applier for the families of the variable elimination backend
  (``"factor_oracle"`` and ``"smoek"``, the latter via
  :func:`~conin.constraints.smoek.create_oracle_constraints_from_smoek`).
  It materialises each constraint into an indicator factor with
  :func:`~conin.representations.variable_elimination.materialise_factor_oracle`;
- :class:`VariableEliminationRepresentation`: the base class of the
  MN/BN/DBN/HMM ``"variable_elimination"`` representations.
"""

import munch

from conin.constraints.smoek import create_oracle_constraints_from_smoek
from conin.graphical_models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from conin.representations.base import Representation, apply_constraints
from conin.util.constraint_data import merge_constraint_data

from .materialise import materialise_factor_oracle


def apply_variable_elimination_constraints(*, pgm, constraints, data=None):
    """The indicator factors of constraints of the variable elimination backend.

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The (possibly unrolled) model the constraints refer to.
    constraints : sequence of FactorOracleConstraint or SmoekConstraint
        Constraints of one family.  Smoek constraints are first combined
        into one :class:`FactorOracleConstraint`.
    data : object, optional
        Passed to the constraint functions (and to callable node scopes).

    Returns
    -------
    list of DiscreteFactor
        One factor per (factor-oracle) constraint, over its scope, ``1`` on
        feasible and ``0`` on infeasible assignments; empty without
        constraints.

    Raises
    ------
    TypeError
        For constraints of several families or of another family.
    ImportError
        For smoek constraints without smoek.
    """
    result = apply_constraints(
        constraints,
        {
            "factor_oracle": lambda cons: _materialise_all(pgm, cons, data),
            "smoek": lambda cons: _materialise_all(
                pgm,
                create_oracle_constraints_from_smoek(
                    pgm=pgm, constraints=cons, data=data
                ),
                data,
            ),
        },
        backend="variable_elimination",
    )
    return [] if result is None else result


def _materialise_all(pgm, constraints, data):
    return [materialise_factor_oracle(con, pgm, data) for con in constraints]


def factor_model(pgm, factors=()):
    """The Markov network of ``pgm``'s factors (its CPDs for a BN) and ``factors``.

    An MN without extra factors is returned as it is.

    Raises
    ------
    TypeError
        For a model that is neither a ``DiscreteMarkovNetwork`` nor a
        ``DiscreteBayesianNetwork``.
    """
    if isinstance(pgm, DiscreteMarkovNetwork):
        if not factors:
            return pgm
        model_factors = list(pgm.factors)
    elif isinstance(pgm, DiscreteBayesianNetwork):
        model_factors = [cpd.to_factor() for cpd in pgm.cpds]
    else:
        raise TypeError(f"Unexpected model type {type(pgm)} for variable elimination")
    return DiscreteMarkovNetwork(
        states=dict(pgm.states), factors=[*model_factors, *factors]
    )


class VariableEliminationRepresentation(Representation):
    """The factor model of a constrained MN/BN/DBN/HMM, used by variable elimination.

    The constraints are materialised once, when the representation is built
    (their ``data`` holds no evidence): :attr:`ve_model` is the Markov network
    of :attr:`pgm`'s factors (its CPDs for a Bayesian network) and one
    indicator factor per constraint, ``0`` on the infeasible assignments of
    its scope.  Evidence is not stored, so one representation can serve
    several queries.

    Attributes
    ----------
    pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The model the constraint applier sees: the MN, the BN, or the
        unrolled BN of a DBN/HMM.
    constraints : tuple
        The user constraints (family ``"factor_oracle"`` or ``"smoek"``).
    ve_model : DiscreteMarkovNetwork
        What variable elimination runs on and what ``write_uai_file`` writes;
        :attr:`pgm` itself for a Markov network without constraints.
    nodes : tuple
        The nodes of ``pgm``.
    data : object or None
        The constrained model's application data.  The constraint functions
        see a copy of it with the fields of :meth:`constraint_data` set.
    """

    kind = "variable_elimination"

    def __init__(self, *, pgm, constraints, data=None):
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(pgm.nodes)
        self.data = data
        self.ve_model = factor_model(
            pgm,
            apply_variable_elimination_constraints(
                pgm=pgm,
                constraints=self.constraints,
                data=merge_constraint_data(data, self.constraint_data()),
            ),
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
