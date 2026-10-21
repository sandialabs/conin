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
        E.g. from ``cmodel.representation("factor").create_model()``.
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
        timer.tic("map_query - STOP")

    return result


def _solve_repn(
    repn,
    *,
    variables,
    evidence,
    show_progress,
    timing,
    write_uai_file,
    determine_variables,
):
    """Create the pgmpy model of ``repn`` and solve it.

    ``evidence`` is the query evidence (keyed by the model's nodes); the
    constraint evidence of ``repn`` is added.  With ``determine_variables``,
    ``variables=None`` becomes the nodes of ``repn.ve_model`` that are not
    evidence (the MN/BN paths); otherwise the solver picks them from the pgmpy
    model (the DBN/HMM paths).
    """
    query_evidence = repn.query_evidence(evidence)
    model = repn.create_model(write_uai_file=write_uai_file)

    if timing:
        TicTocTimer().tic("Created PGMPY model")

    if determine_variables and variables is None:
        if query_evidence:
            variables = [n for n in repn.ve_model.nodes if n not in query_evidence]
        else:
            variables = repn.ve_model.nodes
    return solve_variable_elimination_map_query_model(
        model,
        variables=variables,
        evidence=query_evidence,
        show_progress=show_progress,
    )


def _solve_static(
    repn,
    *,
    variables,
    evidence,
    show_progress,
    timing,
    write_uai_file,
    solution_with_evidence,
    determine_variables,
):
    """Solve a MN/BN/DBN query on ``repn``; add the evidence to the solution if asked.

    The evidence is always added when the model has constraints.
    """
    solution_with_evidence = solution_with_evidence or len(repn.constraints) > 0

    def _execute():
        results = _solve_repn(
            repn,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            write_uai_file=write_uai_file,
            determine_variables=determine_variables,
        )
        if solution_with_evidence and evidence:
            results.solution.states.update(evidence)
        return results

    return _with_timing(timing, _execute)
