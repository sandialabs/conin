"""Run pgmpy's variable elimination MAP query and package the result."""

import numpy as np
from pyomo.common.timing import TicTocTimer

from conin.inference.results import map_results, map_solution, no_solution_results
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.inference
    import pgmpy.models


def solve_variable_elimination_map_query_model(
    model, *, variables=None, evidence=None, show_progress=False
):
    """MAP query on a pgmpy model with ``pgmpy.inference.VariableElimination``.

    Parameters
    ----------
    model : pgmpy.models.DiscreteMarkovNetwork or pgmpy.models.DiscreteBayesianNetwork
        E.g. from ``cmodel.representation("pgmpy").create_model()``.
    variables : list, optional
        The query variables (default: every node not in ``evidence``).
    evidence : dict, optional
        ``{node: state}``.
    show_progress : bool, optional
        Passed to pgmpy.

    Returns
    -------
    munch.Munch
        ``solution`` (``states``, a dict; ``log_potential`` and
        ``solver_objective`` are ``None``), ``solutions`` (``[solution]``),
        ``termination_condition`` (``"ok"``) and ``solvetime``; see
        :mod:`conin.inference.results`.  If the evidence (with the constraint
        factors) has probability zero, there is no solution
        (:func:`~conin.inference.results.no_solution_results`).
    """
    infer = pgmpy.inference.VariableElimination(model)
    if variables is None:
        if evidence:
            variables = [node for node in model.nodes if node not in evidence]
        else:
            variables = [node for node in model.nodes]

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    # A distribution that is zero everywhere has no maximum: pgmpy returns an
    # arbitrary assignment then (after a 0/0 for a Bayesian network, hence the
    # errstate), so the result is checked below.
    with np.errstate(invalid="ignore"):
        map_states = infer.map_query(
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
        )
        feasible = _has_positive_probability(
            model, infer, map_states, variables=variables, evidence=evidence
        )
    solvetime = solver_timer.toc(None)
    if not feasible:
        return no_solution_results(solvetime)
    return map_results([map_solution(map_states)], solvetime=solvetime)


def _has_positive_probability(model, infer, map_states, *, variables, evidence):
    """Whether the MAP assignment ``map_states`` has a positive probability.

    If it does not, every assignment has probability zero with ``evidence``
    (the evidence or the constraint factors exclude them all): the MAP
    assignment is a maximum.  A total assignment is checked factor by
    factor; a partial MAP query (``variables`` not every non-evidence node)
    needs the maximum of the joint distribution, a second elimination.
    """
    assignment = {**(evidence or {}), **map_states}
    if all(node in assignment for node in model.nodes):
        factors = (
            model.get_cpds()
            if isinstance(model, pgmpy.models.DiscreteBayesianNetwork)
            else model.get_factors()
        )
        return all(_value(factor, assignment) > 0 for factor in factors)
    # Over every non-evidence node: pgmpy's max_marginal with no variables
    # is not the maximum of the joint distribution.
    best = infer.max_marginal(
        variables=[node for node in model.nodes if node not in (evidence or {})],
        evidence=evidence,
        show_progress=False,
    )
    return bool(best > 0)  # False for 0 and for NaN (0/0)


def _value(factor, assignment):
    """The value of a pgmpy factor (or CPD) at ``assignment``."""
    index = tuple(factor.name_to_no[var][assignment[var]] for var in factor.variables)
    return factor.values[index]


def _with_timing(timing, func):
    """Call ``func()``, printing START/STOP timer lines if ``timing``."""
    if timing:
        timer = TicTocTimer()
        timer.tic("map_query - START")

    result = func()

    if timing:
        timer.toc("map_query - STOP")

    return result


def _solve_repn(
    repn, *, variables, evidence, show_progress, write_uai_file, timing=False
):
    """Create the pgmpy model of ``repn`` and solve it.

    ``evidence`` is the query evidence (keyed by the model's nodes); the
    constraint evidence of ``repn`` is added.  ``variables=None`` queries
    every node of the pgmpy model that is not evidence.
    """
    model = repn.create_model(write_uai_file=write_uai_file)
    if timing:
        TicTocTimer().tic("Created PGMPY model")
    return solve_variable_elimination_map_query_model(
        model,
        variables=variables,
        evidence=repn.query_evidence(evidence),
        show_progress=show_progress,
    )


def _solve_variable_elimination(repn, q):
    """Solve a MAP query on a ``"pgmpy"`` representation of an MN, BN or DBN.

    ``q`` is the query of :func:`~conin.inference.map_query` (``variables``,
    ``evidence``, ``show_progress``, ``timing`` and ``options``).  The options
    are ``write_uai_file`` (save the factor model, with the constraint
    factors, to this file) and ``solution_with_evidence`` (add the evidence to
    ``solution.states``; always done when the model has constraints); the
    other options are ignored.  An HMM is solved as its unrolled DBN
    (:func:`~conin.inference.hmm_results.solve_unrolled_hmm`).
    """
    solution_with_evidence = (
        q.options.get("solution_with_evidence", False) or len(repn.constraints) > 0
    )

    def _execute():
        results = _solve_repn(
            repn,
            variables=q.variables,
            evidence=q.evidence,
            show_progress=q.show_progress,
            write_uai_file=q.options.get("write_uai_file"),
            timing=q.timing,
        )
        if solution_with_evidence and q.evidence:
            for soln in results.solutions:
                soln.states.update(q.evidence)
        return results

    return _with_timing(q.timing, _execute)
