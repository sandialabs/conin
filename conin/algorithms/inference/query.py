"""The ``map_query`` entry point.

A query goes through four steps:

1. normalise the model: a bare graphical model is wrapped as
   ``Constrained*(model)`` (no constraints), so every query runs on a
   constrained model;
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

from ovld import ovld
from pyomo.common.timing import TicTocTimer

from conin.constrained_models import (
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constrained_models.normalise import as_constrained_model
from conin.exceptions import InfeasibleQueryError, InvalidInputError
from conin.graphical_models.assignments import as_sequence
from conin.graphical_models.dynamic_bayesian_network import check_unroll_start
from conin.representations.matvec import MatVecHMMRepresentation
from conin.representations.mvr import MVRHMMRepresentation
from conin.representations.oracle import OracleHMMRepresentation
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
from conin.representations.variable_elimination import (
    VariableEliminationHMMRepresentation,
    VariableEliminationRepresentation,
)

from .hmm_results import key_solutions_by_time, solve_unrolled_hmm
from .results import (
    map_results,
    map_solution,
    no_solution_results,
    score_hmm_solutions,
    score_solutions,
)
from conin.representations.methods import (
    MAP_METHODS,
    check_method,
    reject_unexpected_options,
    representation_kind,
)
from .oracle.a_star_hmm import _a_star
from .pyomo.hmm_network_flow import _solve_network_flow
from .pyomo.solve import _solve_pyomo
from .query_checks import map_query_args
from .recursive.a_star_hmm import _recursive_a_star
from .toulbar2.solve import _solve_toulbar2
from .unconstrained.viterbi_hmm import _viterbi
from .variable_elimination.solve import _solve_variable_elimination

METHODS = MAP_METHODS
"""The ``method`` values of :func:`map_query`."""

# ---------------------------------------------------------------------
# Steps 1-3: normalise, pick the representation kind, build it
# ---------------------------------------------------------------------


def _representation_kind(method, cmodel, options):
    """Return the representation kind that ``method`` solves on ``cmodel``.

    See :mod:`conin.representations.methods`: ``"integer_program"`` pops
    ``ip_formulation`` from ``options`` (``None``/``"markov_network"`` ->
    ``"pyomo"``, ``"network_flow"`` -> ``"pyomo_network_flow"``), and
    ``"a_star"`` uses ``"recursive"`` for recursive constraints and
    ``"oracle"`` otherwise.

    Raises
    ------
    InvalidInputError
        For an unknown ``ip_formulation``.
    """
    return representation_kind(method, cmodel, options)


def _structure_params(kind, cmodel, *, start, stop):
    """The structure parameters of the ``kind`` representation for this query.

    DBNs are unrolled over ``start..stop`` (``stop=None`` means ``1``, the
    default of the DBN representations).  HMMs are unrolled over
    ``start..stop`` (``stop`` is already set: see :func:`_hmm_time_range`).
    The other kinds have no structure parameters here.
    """
    if kind not in ("pyomo", "toulbar2", "variable_elimination"):
        return {}
    if isinstance(cmodel, ConstrainedDynamicDiscreteBayesianNetwork):
        start, stop = dbn_time_range(start, stop)
        return dict(start=start, stop=stop)
    if isinstance(cmodel, ConstrainedHiddenMarkovModel):
        return dict(start=start, stop=stop)
    return {}


def _observed_time_range(observed, start, stop):
    """The times ``(start, stop)`` of a query on the observations ``observed`` (a list).

    ``stop`` defaults to the last observation, ``len(observed) - 1``.

    Raises
    ------
    InvalidInputError
        If ``stop`` is past the last observation.
    """
    last = len(observed) - 1
    if stop is None:
        return start, last
    if stop > last:
        raise InvalidInputError(
            f"stop={stop} is past the last observation (time {last})"
        )
    return start, stop


def _hmm_observations(method, evidence):
    """Check the observations of an HMM query; return them as a list.

    The observations are a list or tuple, or a dict keyed by the times
    ``0..T-1``.  ``mvr_viterbi`` takes any ``{time: label}`` dict as sparse
    observations (it returns ``None`` for a dict: the MVR Viterbi reads the
    evidence itself).

    Raises
    ------
    InvalidInputError
        For any other evidence.
    """
    if method == "mvr_viterbi" and isinstance(evidence, dict):
        return None
    what = f"evidence (the observations) of map_query(method={method!r}) on an HMM"
    return as_sequence(evidence, what)


def _hmm_time_range(observed, start, stop):
    """The times ``(start, stop)`` and the observations of an HMM query.

    Every HMM method infers over ``start..stop``: ``stop`` defaults to the
    last observation, and the observations after ``stop`` are dropped.
    ``observed`` is ``None`` for the sparse observations of ``mvr_viterbi``,
    whose ``stop`` may also be after the last observation.

    Raises
    ------
    InvalidInputError
        If ``stop`` is past the last (list) observation.
    """
    if observed is None:
        return start, stop, None
    start, stop = _observed_time_range(observed, start, stop)
    return start, stop, observed[: stop + 1]


def map_query(
    pgm,
    *,
    method,
    variables=None,
    evidence=None,
    timing=False,
    start=0,
    stop=None,
    **options,
):
    """Compute a MAP assignment

    The model is normalised to a constrained model (a bare model is used with
    no constraints), ``method`` selects a representation of it (see the table
    below), and the query is solved on that representation.

    ======================== =========================================== =======================
    ``method``               representation kind                         models
    ======================== =========================================== =======================
    ``integer_program``      ``"pyomo"``, or ``"pyomo_network_flow"``    MN, BN, DBN, HMM
                             with ``ip_formulation="network_flow"``      (network flow: HMM)
    ``toulbar2``             ``"toulbar2"``                              MN, BN, DBN, HMM
    ``variable_elimination`` ``"variable_elimination"``                  MN, BN, DBN, HMM
    ``a_star``               ``"oracle"``, or ``"recursive"`` for        HMM
                             recursive constraints
    ``viterbi``              ``"matvec"`` (no constraints)               HMM
    ``mvr_viterbi``          ``"mvr"`` (needs torch)                     HMM
    ======================== =========================================== =======================

    Parameters
    ----------
    pgm : Graphical model to solve.
        A CONIN model or constrained model.  Convert a pgmpy model with
        :func:`conin.io.pgmpy.convert_pgmpy_to_conin` first.
    method : string
        The string name of the inference method.
    variables : list, optional
        Variables included in the MAP query. Support for partial MAP queries
        depends on the selected method: ``variable_elimination`` supports
        them; the other methods raise ``NotImplementedError`` unless
        ``variables`` names every node.
    evidence : dict or list, optional
        Observed variable assignments. For static models, a dictionary of
        ``{variable: state}``. For dynamic Bayesian networks, a dictionary
        keyed by ``(variable, time)``. For hidden Markov models, the
        observations (required): a list or tuple, or a dictionary keyed by
        the times ``0..T-1`` (the hidden states are then returned keyed by
        time index); ``mvr_viterbi`` also accepts a sparse ``{time: label}``
        dictionary.
    timing : bool, optional
        If ``True``, print timing information.
        Not used by every method.
    start : int, optional
        Initial time index for dynamic models (ignored by static models).
        Only ``0`` is supported: any other value raises
        ``InvalidInputError``.
    stop : int, optional
        Final time index for dynamic models (ignored by static models).
        For DBNs it defaults to ``1``.  For HMMs, every method infers over
        ``start..stop`` and ignores the observations after ``stop``, which
        defaults to the last observation (a later ``stop`` raises
        ``InvalidInputError``).  With sparse ``{time: label}`` observations,
        ``mvr_viterbi`` infers over ``0..stop``, where ``stop`` defaults to
        the last observed time and may be after it.
    options : dict, optional
        Additional keyword arguments forwarded to the selected
        inference helper, such as solver options or output-file settings.
        ``integer_program`` takes ``ip_formulation`` (``None``,
        ``"markov_network"`` or ``"network_flow"``); ``mvr_viterbi`` takes
        ``dtype`` and ``device``.  Every method raises ``TypeError`` for an
        option it does not take (``viterbi`` takes none).

    Returns
    -------
    munch.Munch
        The same keys for every method (see :mod:`conin.algorithms.inference.results`):
        ``solution`` (the best solution), ``solutions`` (best first),
        ``termination_condition`` (``"ok"`` or why the search stopped) and
        ``solvetime``.  If the evidence or the constraints exclude every
        assignment, ``solution`` is ``None``, ``solutions`` is empty and
        ``termination_condition`` is ``"error: no feasible solutions"``, for
        every method (``integer_program`` with ``solver="or_topas"``
        raises instead).  Each solution has
        ``states`` (``{node: state}``; for an HMM, the hidden states as a
        list, or a dict keyed by time step if the evidence was a dict),
        ``log_potential`` (:func:`conin.algorithms.scoring.log_potential` of the
        assignment and the evidence on the model without its constraints;
        ``None`` for a partial MAP query) and ``solver_objective`` (the
        integer program's or toulbar2's objective value, in the backend's
        scale; ``None`` for the other methods).  ``mvr_viterbi`` solutions
        also have ``augmented`` (the decoded augmented path).

    Raises
    ------
    InvalidInputError
        For an unknown ``method`` or ``ip_formulation``, if ``method`` does
        not support the model type or its constraint family (no such
        representation is registered), if the evidence of an HMM is not a
        list, tuple or dict keyed by ``0..T-1``, if ``start`` is not ``0``
        for a DBN or HMM, or if ``stop`` is past the last observation.
    TypeError
        If ``pgm`` is not a supported model type, or for an option that the
        method does not take.
    NotImplementedError
        For a partial MAP query (``variables``) with a method other than
        ``variable_elimination``.
    ImportError
        If the method's backend is not installed (pytoulbar2 for
        ``toulbar2``, torch for ``mvr_viterbi``).

    Examples
    --------
    Static model (Markov network):

    >>> import conin.examples.markov_network
    >>> from conin.algorithms.inference import map_query
    >>> example = conin.examples.markov_network.ABC_conin()
    >>> results = map_query(example.pgm, method="integer_program", solver="glpk")  # doctest: +SKIP

    Dynamic model (Hidden Markov model):

    >>> import conin.examples.hidden_markov_model
    >>> from conin.algorithms.inference import map_query
    >>> pgm = conin.examples.hidden_markov_model.create_hmm1()
    >>> results = map_query(pgm, method="integer_program", evidence=["o0", "o1"], solver="glpk")  # doctest: +SKIP
    """
    check_method(method, METHODS, "inference type")
    cmodel = as_constrained_model(pgm)
    observed = None
    if isinstance(
        cmodel,
        (ConstrainedDynamicDiscreteBayesianNetwork, ConstrainedHiddenMarkovModel),
    ):
        check_unroll_start(start)
    if isinstance(cmodel, ConstrainedHiddenMarkovModel):
        observed = _hmm_observations(method, evidence)
        start, stop, observed = _hmm_time_range(observed, start, stop)
    kind = _representation_kind(method, cmodel, options)
    repn = cmodel.representation(
        kind, **_structure_params(kind, cmodel, start=start, stop=stop)
    )
    query = map_query_args(
        model=cmodel.model,
        variables=variables,
        evidence=evidence,
        observed=observed,
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
# The factor backends (Pyomo, toulbar2, variable elimination) have one solver each, for the
# MN, BN and DBN representations (overloads on the backend's base class).
# Their HMM representations are solved as the unrolled DBN by
# solve_unrolled_hmm.  Network flow, Viterbi, A* and the MVR Viterbi are
# HMM algorithms with their own overloads.  Every result has the shape of
# conin.algorithms.inference.results: the factor backends' solutions are scored here,
# the HMM algorithms score their own.


def _factor_solve(solve, repn, q):
    """Solve an MN, BN or DBN query; score the solutions on the (unrolled) model."""
    results = solve(repn, q)
    return score_solutions(results, repn.pgm, evidence=q.evidence, nodes=repn.nodes)


def _unrolled_hmm_solve(solve, repn, q):
    """Solve an HMM query as its unrolled DBN; score the solutions on the HMM."""
    results = solve_unrolled_hmm(solve, repn, q)
    return score_hmm_solutions(results, q.model, q.observed)


@ovld
def _solve_map_query(repn: PyomoFactorRepresentation, q):
    return _factor_solve(_solve_pyomo, repn, q)


@ovld
def _solve_map_query(repn: PyomoHMMRepresentation, q):
    return _unrolled_hmm_solve(_solve_pyomo, repn, q)


@ovld
def _solve_map_query(repn: PyomoNetworkFlowHMMRepresentation, q):
    return _solve_network_flow(repn, q)


@ovld
def _solve_map_query(repn: Toulbar2FactorRepresentation, q):
    return _factor_solve(_solve_toulbar2, repn, q)


@ovld
def _solve_map_query(repn: Toulbar2HMMRepresentation, q):
    return _unrolled_hmm_solve(_solve_toulbar2, repn, q)


@ovld
def _solve_map_query(repn: VariableEliminationRepresentation, q):
    return _factor_solve(_solve_variable_elimination, repn, q)


@ovld
def _solve_map_query(repn: VariableEliminationHMMRepresentation, q):
    return _unrolled_hmm_solve(_solve_variable_elimination, repn, q)


def _hmm_search(search, repn, q, **options):
    """Run an HMM search on the observations; key the solutions by time for dict evidence."""
    results = search(repn, observed=q.observed, **options)
    if isinstance(q.evidence, dict):
        key_solutions_by_time(results)
    return results


@ovld
def _solve_map_query(repn: MatVecHMMRepresentation, q):
    reject_unexpected_options(q.options, "map_query(method='viterbi')")
    return _hmm_search(_viterbi, repn, q)


@ovld
def _solve_map_query(repn: OracleHMMRepresentation, q):
    return _hmm_search(_a_star, repn, q, **q.options)


@ovld
def _solve_map_query(repn: RecursiveHMMRepresentation, q):
    return _hmm_search(_recursive_a_star, repn, q, **q.options)


@ovld
def _solve_map_query(repn: MVRHMMRepresentation, q):
    # List observations are already cut at stop (q.observed); a dict is sparse.
    evidence = q.evidence if q.observed is None else q.observed
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
    from conin.algorithms.inference.mvr.viterbi import _viterbi_torch_mvr

    timer = TicTocTimer()
    timer.tic(None)
    try:
        hidden, augmented = _viterbi_torch_mvr(
            repn,
            evidence,
            time_horizon=time_horizon,
            return_augmented=True,
            **q.options,
        )
    except InfeasibleQueryError:
        return no_solution_results(timer.toc(None))
    solvetime = timer.toc(None)
    states = dict(enumerate(hidden)) if isinstance(evidence, dict) else hidden
    results = map_results(
        [map_solution(states, augmented=augmented)], solvetime=solvetime
    )
    # The observations may be sparse ({time: label}); an unobserved time
    # contributes its transition only, as in the MVR Viterbi's own score.
    return score_hmm_solutions(results, repn.model, evidence)
