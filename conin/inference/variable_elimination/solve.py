"""Run pgmpy's variable elimination MAP query and package the result."""

import munch
from pyomo.common.timing import TicTocTimer

from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.inference


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
        ``solution.states`` (a dict) and ``solvetime``.
    """
    infer = pgmpy.inference.VariableElimination(model)
    if variables is None:
        if evidence:
            variables = [node for node in model.nodes if node not in evidence]
        else:
            variables = [node for node in model.nodes]

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    map_states = infer.map_query(
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
    )
    solvetime = solver_timer.toc(None)
    return munch.Munch(solution=munch.Munch(states=map_states), solvetime=solvetime)


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
            results.solution.states.update(q.evidence)
        return results

    return _with_timing(q.timing, _execute)
