"""Run a variable elimination MAP query and package the result."""

import math

from pyomo.common.timing import TicTocTimer

from conin.algorithms.inference.results import (
    map_results,
    map_solution,
    no_solution_results,
)
from conin.algorithms.util.variable_elimination import max_product
from conin.io import save_model
from conin.representations.methods import reject_unexpected_options


def solve_variable_elimination_map_query_model(model, *, variables=None, evidence=None):
    """MAP query on a Markov network with variable elimination.

    Parameters
    ----------
    model : DiscreteMarkovNetwork
        E.g. ``cmodel.representation("variable_elimination").ve_model``.
    variables : list, optional
        The query variables (default: every node not in ``evidence``).  The
        other nodes that are not evidence are summed out (a partial MAP
        query).
    evidence : dict, optional
        ``{node: state}``.

    Returns
    -------
    munch.Munch
        ``solution`` (``states``, a dict; ``log_potential`` and
        ``solver_objective`` are ``None``), ``solutions`` (``[solution]``),
        ``termination_condition`` (``"ok"``) and ``solvetime``; see
        :mod:`conin.algorithms.inference.results`.  If the evidence (with the
        constraint factors) has probability zero, there is no solution
        (:func:`~conin.algorithms.inference.results.no_solution_results`).

    Raises
    ------
    InvalidInputError
        For an unknown variable or state, or a variable that is also
        evidence.
    """
    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    value, states = max_product(
        model.factors, model.states, variables=variables, evidence=evidence
    )
    solvetime = solver_timer.toc(None)
    if value == -math.inf:
        return no_solution_results(solvetime)
    return map_results([map_solution(states)], solvetime=solvetime)


def _with_timing(timing, func):
    """Call ``func()``, printing START/STOP timer lines if ``timing``."""
    if timing:
        timer = TicTocTimer()
        timer.tic("map_query - START")

    result = func()

    if timing:
        timer.toc("map_query - STOP")

    return result


def _solve_variable_elimination(repn, q):
    """Solve a MAP query on a ``"variable_elimination"`` representation of an MN, BN or DBN.

    ``q`` is the query of :func:`~conin.algorithms.inference.map_query` (``variables``,
    ``evidence``, ``timing`` and ``options``).  The options are
    ``write_uai_file`` (save the factor model, with the constraint factors,
    to this file), ``solution_with_evidence`` and its synonym
    ``solution_with_fixed`` (add the evidence to ``solution.states``; always
    done when the model has constraints).  Other options raise
    ``TypeError``.  An HMM is
    solved as its unrolled DBN
    (:func:`~conin.algorithms.inference.hmm_results.solve_unrolled_hmm`).
    """
    options = dict(q.options)
    write_uai_file = options.pop("write_uai_file", None)
    with_evidence = options.pop("solution_with_evidence", False)
    with_fixed = options.pop("solution_with_fixed", False)
    solution_with_evidence = with_evidence or with_fixed or len(repn.constraints) > 0
    reject_unexpected_options(options, "map_query(method='variable_elimination')")

    def _execute():
        if write_uai_file:
            save_model(repn.ve_model, write_uai_file)
        results = solve_variable_elimination_map_query_model(
            repn.ve_model, variables=q.variables, evidence=q.evidence
        )
        if solution_with_evidence and q.evidence:
            for soln in results.solutions:
                soln.states.update(q.evidence)
        return results

    return _with_timing(q.timing, _execute)
