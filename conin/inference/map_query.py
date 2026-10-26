"""The ``map_query`` entry point.

A query goes through four steps:

1. normalise the model: a pgmpy model is converted to a CONIN model, and a
   bare graphical model is wrapped as ``Constrained*(model)`` (no
   constraints), so every query runs on a constrained model;
2. pick the representation kind from ``method`` (and, for
   ``"integer_program"``, ``ip_formulation``; for ``"a_star"``, the
   constraint family);
3. build that representation with its structure parameters
   (``cmodel.representation(kind, **params)``);
4. solve the query with the overload of :func:`_solve_map_query` for the
   representation's class.

Whether a (model type, constraint family) pair supports a method is decided
by the representation registry: unsupported pairs raise its
``InvalidInputError``.
"""

import munch
from ovld import ovld

from conin.constrained_models import (
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.normalise import as_constrained_model
from conin.exceptions import InvalidInputError
from conin.graphical_models.hmm_observations import (
    as_sequence,
    require_sequence_type,
    time_range,
)
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.mvr import MVRHMMRepresentation
from conin.representations.oracle import OracleHMMRepresentation
from conin.representations.pgmpy import PgmpyHMMRepresentation, PgmpyRepresentation
from conin.representations.pyomo import (
    PyomoFactorRepresentation,
    PyomoHMMRepresentation,
    PyomoNetworkFlowHMMRepresentation,
)
from conin.representations.recursive import RecursiveHMMRepresentation
from conin.representations.toulbar2 import (
    Toulbar2FactorRepresentation,
    Toulbar2HMMRepresentation,
)
from conin.representations.unrolled import dbn_time_range

from .hmm_results import key_solutions_by_time, solve_unrolled_hmm
from .oracle.a_star_hmm import _a_star
from .pyomo.hmm_network_flow import _solve_network_flow
from .pyomo.solve import _solve_pyomo
from .query_checks import map_query_args
from .recursive.a_star_hmm import _recursive_a_star
from .toulbar2.solve import _solve_toulbar2
from .unconstrained.viterbi import _viterbi
from .variable_elimination.solve import _solve_variable_elimination

METHODS = (
    "integer_program",
    "toulbar2",
    "a_star",
    "viterbi",
    "variable_elimination",
    "mvr_viterbi",
)
"""The ``method`` values of :func:`map_query`."""

# ---------------------------------------------------------------------
# Steps 1-3: normalise, pick the representation kind, build it
# ---------------------------------------------------------------------


def _representation_kind(method, cmodel, options):
    """Return the representation kind that ``method`` solves on ``cmodel``.

    ``"integer_program"`` pops ``ip_formulation`` from ``options``:
    ``None``/``"markov_network"`` -> ``"pyomo"``, ``"network_flow"`` ->
    ``"pyomo_network_flow"``.  ``"a_star"`` uses ``"recursive"`` for
    recursive constraints and ``"oracle"`` otherwise.

    Raises
    ------
    ValueError
        For an unknown ``ip_formulation``.
    """
    if method == "integer_program":
        ip_formulation = options.pop("ip_formulation", None)
        if ip_formulation is None or ip_formulation == "markov_network":
            return "pyomo"
        if ip_formulation == "network_flow":
            return "pyomo_network_flow"
        raise ValueError(f"Unexpected ip_formulation value: '{ip_formulation}'")
    if method == "a_star":
        return "recursive" if cmodel.constraint_family == "recursive" else "oracle"
    return {
        "toulbar2": "toulbar2",
        "variable_elimination": "pgmpy",
        "viterbi": "matvec",
        "mvr_viterbi": "mvr",
    }[method]


def _structure_params(kind, cmodel, *, observed, start, stop):
    """The structure parameters of the ``kind`` representation for this query.

    DBNs are unrolled over ``start..stop`` (``stop=None`` means ``1``, the
    default of the DBN representations).  HMMs
    are unrolled over ``start..stop``, where ``observed`` is the list of
    observations and ``stop`` defaults to the last one (a later ``stop``
    raises ``InvalidInputError``).  The other kinds have no structure
    parameters here.
    """
    if kind not in ("pyomo", "toulbar2", "pgmpy"):
        return {}
    if isinstance(cmodel, ConstrainedDynamicDiscreteBayesianNetwork):
        start, stop = dbn_time_range(start, stop)
        return dict(start=start, stop=stop)
    if isinstance(cmodel, ConstrainedHiddenMarkovModel):
        start, stop = time_range(observed, start, stop)
        return dict(start=start, stop=stop)
    return {}


def _hmm_observations(method, evidence):
    """Check the observations of an HMM query; return them as a list.

    The observations are a list or tuple, or a dict keyed by the times
    ``0..T-1``.  ``mvr_viterbi`` also accepts a sparse ``{time: label}`` dict
    (it returns ``None`` for it: the MVR Viterbi reads the evidence itself).

    Raises
    ------
    InvalidInputError
        For any other evidence.
    """
    what = f"evidence (the observations) of map_query(method={method!r}) on an HMM"
    if method == "mvr_viterbi":
        require_sequence_type(evidence, what)
        return None
    return as_sequence(evidence, what)


def map_query(
    pgm,
    *,
    method,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    **options,
):
    """Compute a MAP assignment

    The model is normalised to a constrained model (pgmpy models are
    converted; a bare model is used with no constraints), ``method`` selects a
    representation of it (see the table below), and the query is solved on
    that representation.

    ======================== =========================================== =======================
    ``method``               representation kind                         models
    ======================== =========================================== =======================
    ``integer_program``      ``"pyomo"``, or ``"pyomo_network_flow"``    MN, BN, DBN, HMM
                             with ``ip_formulation="network_flow"``      (network flow: HMM)
    ``toulbar2``             ``"toulbar2"``                              MN, BN, DBN, HMM
    ``variable_elimination`` ``"pgmpy"``                                 MN, BN, DBN, HMM
    ``a_star``               ``"oracle"``, or ``"recursive"`` for        HMM
                             recursive constraints
    ``viterbi``              ``"matvec"`` (no constraints)               HMM
    ``mvr_viterbi``          ``"mvr"`` (needs torch)                     HMM
    ======================== =========================================== =======================

    Parameters
    ----------
    pgm : Graphical model to solve.
        A CONIN model or constrained model.  Compatible pgmpy models are
        converted to CONIN model objects if pgmpy is installed.
    method : string
        The string name of the inference method.
    variables : list, optional
        Variables included in the MAP query. Support for partial MAP queries
        depends on the selected method: ``variable_elimination`` supports
        them; ``integer_program`` and ``toulbar2`` raise ``RuntimeError``
        unless ``variables`` names every node.
    evidence : dict or list, optional
        Observed variable assignments. For static models, a dictionary of
        ``{variable: state}``. For dynamic Bayesian networks, a dictionary
        keyed by ``(variable, time)``. For hidden Markov models, the
        observations (required): a list or tuple, or a dictionary keyed by
        the times ``0..T-1`` (the hidden states are then returned keyed by
        time index); ``mvr_viterbi`` also accepts a sparse ``{time: label}``
        dictionary.
    show_progress : bool, optional
        Whether to request solver progress reporting when supported.
    timing : bool, optional
        If ``True``, print timing information.
        Not used by every method.
    start : int, optional
        Initial time index for dynamic models (ignored by static models).
    stop : int, optional
        Final time index for dynamic models (ignored by static models).
        For DBNs it defaults to ``1``.  For HMMs, ``integer_program``,
        ``toulbar2`` and ``variable_elimination`` infer over ``start..stop``,
        with ``stop`` defaulting to ``len(evidence) - 1`` (a later ``stop``
        raises ``InvalidInputError``); ``viterbi`` and ``a_star`` use every
        observation.  For ``mvr_viterbi`` the horizon is ``stop + 1`` time
        steps.
    options : dict, optional
        Additional keyword arguments forwarded to the selected
        inference helper, such as solver options or output-file settings.
        ``integer_program`` takes ``ip_formulation`` (``None``,
        ``"markov_network"`` or ``"network_flow"``); ``mvr_viterbi`` takes
        ``dtype`` and ``device``.

    Returns
    -------
    munch.Munch
        Result object produced by the selected inference helper. The
        returned object contains ``solution`` (with ``states``; ``None`` if
        A* finds no solution), ``solutions`` and, for the solver-based
        methods, ``solvetime``. The HMM methods also return ``observed``;
        ``mvr_viterbi`` solutions also have ``log_likelihood`` and
        ``augmented`` (the decoded augmented path).

    Raises
    ------
    ValueError
        For an unknown ``method`` or ``ip_formulation``.
    TypeError
        If ``pgm`` is not a supported model type.
    InvalidInputError
        If ``method`` does not support the model type or its constraint
        family (no such representation is registered), if the evidence of
        an HMM is not a list, tuple or dict keyed by ``0..T-1``, or if
        ``stop`` is past the last observation.

    Examples
    --------
    Static model (Markov network):

    >>> import conin.examples.markov_network
    >>> from conin.inference import map_query
    >>> example = conin.examples.markov_network.ABC_conin()
    >>> results = map_query(example.pgm, method="integer_program", solver="glpk")  # doctest: +SKIP

    Dynamic model (Hidden Markov model):

    >>> import conin.examples.hidden_markov_model
    >>> from conin.inference import map_query
    >>> pgm = conin.examples.hidden_markov_model.create_hmm1()
    >>> results = map_query(pgm, method="integer_program", evidence=["o0", "o1"], solver="glpk")  # doctest: +SKIP
    """
    if method not in METHODS:
        raise ValueError(
            f"Unsupported inference type: {method}. "
            f"Expected one of: {', '.join(METHODS)}"
        )
    cmodel = as_constrained_model(pgm)
    observed = None
    if isinstance(cmodel, ConstrainedHiddenMarkovModel):
        observed = _hmm_observations(method, evidence)
    kind = _representation_kind(method, cmodel, options)
    repn = cmodel.representation(
        kind,
        **_structure_params(kind, cmodel, observed=observed, start=start, stop=stop),
    )
    query = map_query_args(
        variables=variables,
        evidence=evidence,
        observed=observed,
        show_progress=show_progress,
        timing=timing,
        start=start,
        stop=stop,
        options=options,
    )
    return _solve_map_query(repn, query)


# ---------------------------------------------------------------------
# Step 4: one overload per backend, and one per HMM algorithm
# ---------------------------------------------------------------------
#
# The factor backends (Pyomo, toulbar2, pgmpy) have one solver each, for the
# MN, BN and DBN representations (overloads on the backend's base class).
# Their HMM representations are solved as the unrolled DBN by
# solve_unrolled_hmm.  Network flow, Viterbi, A* and the MVR Viterbi are
# HMM algorithms with their own overloads.


@ovld
def _solve_map_query(repn: PyomoFactorRepresentation, q):
    return _solve_pyomo(repn, q)


@ovld
def _solve_map_query(repn: PyomoHMMRepresentation, q):
    return solve_unrolled_hmm(_solve_pyomo, repn, q)


@ovld
def _solve_map_query(repn: PyomoNetworkFlowHMMRepresentation, q):
    return _solve_network_flow(repn, q)


@ovld
def _solve_map_query(repn: Toulbar2FactorRepresentation, q):
    return _solve_toulbar2(repn, q)


@ovld
def _solve_map_query(repn: Toulbar2HMMRepresentation, q):
    return solve_unrolled_hmm(_solve_toulbar2, repn, q)


@ovld
def _solve_map_query(repn: PgmpyRepresentation, q):
    return _solve_variable_elimination(repn, q)


@ovld
def _solve_map_query(repn: PgmpyHMMRepresentation, q):
    return solve_unrolled_hmm(_solve_variable_elimination, repn, q)


def _hmm_search(search, repn, q, **options):
    """Run an HMM search on the observations; key the solutions by time for dict evidence."""
    results = search(repn, observed=q.observed, **options)
    if isinstance(q.evidence, dict):
        key_solutions_by_time(results)
    return results


@ovld
def _solve_map_query(repn: MatVecHMMRepresentation, q):
    # Viterbi has no options; the other options are ignored.
    return _hmm_search(_viterbi, repn, q)


@ovld
def _solve_map_query(repn: OracleHMMRepresentation, q):
    return _hmm_search(_a_star, repn, q, **q.options)


@ovld
def _solve_map_query(repn: RecursiveHMMRepresentation, q):
    results = _hmm_search(_recursive_a_star_result, repn, q, **q.options)
    results.solution = results.solutions[0] if results.solutions else None
    return results


def _recursive_a_star_result(repn, *, observed, **options):
    """Recursive A* with solutions shaped like A*'s (``states``, ``hidden``, ``log_likelihood``)."""
    results = _recursive_a_star(repn, observed=observed, **options)
    for soln in results.solutions:
        soln.states = soln.hidden
    return results


@ovld
def _solve_map_query(repn: MVRHMMRepresentation, q):
    if q.start != 0:
        raise InvalidInputError(
            f"mvr_viterbi infers over the times 0..stop; start must be 0, got {q.start}"
        )
    evidence = q.evidence
    if q.stop is not None:
        time_horizon = q.stop + 1
    elif isinstance(evidence, dict):
        if not evidence:
            raise InvalidInputError(
                "mvr_viterbi needs a stop time or at least one observation"
            )
        time_horizon = max(evidence) + 1
    else:
        time_horizon = None
    # Imported here so that importing conin does not import torch.
    from conin.inference.mvr.viterbi import _viterbi_torch_mvr

    hidden, augmented, log_likelihood = _viterbi_torch_mvr(
        repn,
        evidence,
        time_horizon=time_horizon,
        return_augmented=True,
        return_score=True,
        **q.options,
    )
    states = dict(enumerate(hidden)) if isinstance(evidence, dict) else hidden
    soln = munch.Munch(
        states=states,
        hidden=states,
        log_likelihood=log_likelihood,
        augmented=augmented,
    )
    return munch.Munch(
        observed=evidence,
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
    )
