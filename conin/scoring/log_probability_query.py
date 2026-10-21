"""The ``log_probability_query`` entry point: marginal and conditional log-probabilities.

A query goes through the same steps as ``map_query``:

1. normalise the model (pgmpy models are converted; a bare model is wrapped
   with no constraints);
2. pick the representation kind from ``method`` (only
   ``"variable_elimination"`` -> ``"factor"`` for now);
3. build that representation with its structure parameters (DBNs and HMMs
   are unrolled over ``start..stop``);
4. compute the query with the overload of :func:`_solve_log_probability` for
   the representation's class.

Unsupported (model type, constraint family) pairs raise the representation
registry's ``InvalidInputError``.
"""

import numbers

import munch
from ovld import ovld

from conin.constrained_models import (
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.normalise import as_constrained_model
from conin.exceptions import InvalidInputError
from conin.representations.factor import (
    FactorBNRepresentation,
    FactorDBNRepresentation,
    FactorHMMRepresentation,
    FactorMNRepresentation,
)

from .variable_elimination import _log_probability_variable_elimination

METHODS = ("variable_elimination",)
"""The ``method`` values of :func:`log_probability_query`."""


def _representation_kind(method):
    """Return the representation kind that ``method`` computes the query on."""
    return {"variable_elimination": "factor"}[method]


def _structure_params(cmodel, *, variables, start, stop):
    """The structure parameters of the query's representation.

    DBNs and HMMs are unrolled over ``start..stop``; ``stop`` defaults to the
    largest time of the variables ``(name, t)`` in the query, and at least
    ``start + 1``.  Static models have no structure parameters.

    Raises
    ------
    InvalidInputError
        If ``stop`` is not after ``start``.
    """
    if not isinstance(
        cmodel,
        (ConstrainedDynamicDiscreteBayesianNetwork, ConstrainedHiddenMarkovModel),
    ):
        return {}
    if stop is None:
        times = [
            var[1]
            for var in variables
            if isinstance(var, tuple)
            and len(var) == 2
            and isinstance(var[1], numbers.Integral)
        ]
        stop = max(times + [start + 1])
    elif stop <= start:
        raise InvalidInputError(
            f"A DBN or HMM is unrolled over start..stop with start < stop; got "
            f"start={start}, stop={stop}"
        )
    return dict(start=start, stop=stop)


def _as_dict(value, what):
    if not isinstance(value, dict):
        raise InvalidInputError(
            f"The {what} of log_probability_query must be a dict, got "
            f"{type(value).__name__}"
        )
    return value


def log_probability_query(
    pgm,
    assignment,
    *,
    evidence=None,
    method="variable_elimination",
    start=0,
    stop=None,
    **options,
):
    """Log-probability of a (partial) assignment, given evidence and the constraints

    Returns ``log P(X = x | E = e)``, where the variables in neither
    ``assignment`` nor ``evidence`` are summed out.  For a constrained model
    it is the probability under the constraints, ``log P(X = x | E = e, C)``
    (``log P(x, e, C) - log P(e, C)``), where ``C`` is the event that every
    constraint holds.  Markov networks are normalised (the partition function
    cancels).

    ======================== ==================== ================
    ``method``               representation kind  models
    ======================== ==================== ================
    ``variable_elimination`` ``"factor"``         MN, BN, DBN, HMM
    ======================== ==================== ================

    Parameters
    ----------
    pgm : GraphicalModel or ConstrainedGraphicalModel
        A CONIN model or constrained model (constraint families ``None``,
        ``"factor_oracle"`` and ``"algebraic"``), or a pgmpy
        ``DiscreteMarkovNetwork``, ``DiscreteBayesianNetwork`` or
        ``DynamicBayesianNetwork`` (converted to CONIN; needs pgmpy).
    assignment : dict
        ``{variable: state}`` for the variables whose probability is
        computed (at least one).  The variables of a DBN are ``(name, t)``;
        those of an HMM are ``("H", t)`` (hidden) and ``("E", t)``
        (observed), as in the HMM unrolled as a DBN.  A total assignment of a
        BN, DBN or HMM gives :func:`~conin.scoring.log_probability`.
    evidence : dict, optional
        ``{variable: state}`` to condition on, keyed as ``assignment``; no
        variable may be in both.
    method : str, optional
        The inference method (default ``"variable_elimination"``, which needs
        pgmpy).
    start, stop : int, optional
        DBNs and HMMs are unrolled over ``start..stop`` (ignored by static
        models).  ``stop`` defaults to the largest time in ``assignment`` and
        ``evidence``, and at least ``start + 1``.  Without constraints a later
        ``stop`` gives the same result; with constraints it sets the horizon
        the constraints see (e.g. the time steps of an HMM constraint).
    options : dict, optional
        Options of the method; ``variable_elimination`` takes none.

    Returns
    -------
    float
        The log-probability; ``-inf`` if it is zero.  ``variable_elimination``
        sums in linear space, so a probability below about ``1e-308`` (e.g.
        a long HMM sequence) underflows to ``-inf``.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    TypeError
        If ``pgm`` is not a supported model type, or for an unknown option.
    InvalidInputError
        If the method does not support the model's constraint family (no
        such representation is registered); for an ``assignment`` or
        ``evidence`` that is not a dict, an empty ``assignment``, a variable
        in both, an unknown variable or state, or ``stop <= start``; or if
        the evidence (with the constraints) has probability zero, so the
        conditional is not defined.

    Examples
    --------
    Variable elimination needs pgmpy:

    >>> from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
    >>> from conin.scoring import log_probability_query
    >>> bn = DiscreteBayesianNetwork(
    ...     states={"A": [0, 1], "B": [0, 1]},
    ...     cpds=[
    ...         DiscreteCPD(node="A", values=[0.9, 0.1]),
    ...         DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [0.9, 0.1]}),
    ...     ],
    ... )
    >>> # log(0.9 * 0.8 + 0.1 * 0.1)
    >>> round(log_probability_query(bn, {"B": 1}), 6)  # doctest: +SKIP
    -0.314711
    >>> # log(0.1 * 0.1 / 0.73)
    >>> round(log_probability_query(bn, {"A": 1}, evidence={"B": 1}), 6)  # doctest: +SKIP
    -4.290459
    """
    if method not in METHODS:
        raise ValueError(
            f"Unsupported log-probability method: {method}. "
            f"Expected one of: {', '.join(METHODS)}"
        )
    cmodel = as_constrained_model(pgm)
    assignment = _as_dict(assignment, "assignment")
    evidence = {} if evidence is None else _as_dict(evidence, "evidence")
    if not assignment:
        raise InvalidInputError(
            "The assignment of log_probability_query must give a state for at "
            "least one variable"
        )
    both = [var for var in assignment if var in evidence]
    if both:
        raise InvalidInputError(
            f"Variables cannot be in both the assignment and the evidence: {both}"
        )
    kind = _representation_kind(method)
    repn = cmodel.representation(
        kind,
        **_structure_params(
            cmodel, variables=[*assignment, *evidence], start=start, stop=stop
        ),
    )
    query = munch.Munch(assignment=assignment, evidence=evidence, options=options)
    return _solve_log_probability(repn, query)


def _check_variables(states, values, what):
    """Raise ``InvalidInputError`` unless ``values`` maps variables of ``states`` to their states."""
    for var, state in values.items():
        if var not in states:
            raise InvalidInputError(
                f"Unknown variable {var!r} in the {what} (the model's "
                f"variables are {list(states)})"
            )
        if state not in states[var]:
            raise InvalidInputError(
                f"Unknown state {state!r} of {var!r} in the {what} (its states "
                f"are {list(states[var])})"
            )


def _variable_elimination(repn, q):
    if q.options:
        raise TypeError(
            "Unexpected options for log_probability_query("
            f"method='variable_elimination'): {', '.join(map(str, q.options))}"
        )
    _check_variables(repn.pgm.states, q.assignment, "assignment")
    _check_variables(repn.pgm.states, q.evidence, "evidence")
    return _log_probability_variable_elimination(
        repn, assignment=q.assignment, evidence=q.evidence
    )


@ovld
def _solve_log_probability(repn: FactorMNRepresentation, q):
    return _variable_elimination(repn, q)


@ovld
def _solve_log_probability(repn: FactorBNRepresentation, q):
    return _variable_elimination(repn, q)


@ovld
def _solve_log_probability(repn: FactorDBNRepresentation, q):
    return _variable_elimination(repn, q)


@ovld
def _solve_log_probability(repn: FactorHMMRepresentation, q):
    return _variable_elimination(repn, q)
