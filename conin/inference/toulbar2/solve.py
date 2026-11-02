"""Solve toulbar2 MAP query models.

The models are built by the ``"toulbar2"`` representations
(:mod:`conin.representations.toulbar2`).
"""

from pyomo.common.timing import TicTocTimer

from conin.inference.query_checks import full_query_variables, reject_pruning_options
from conin.inference.results import map_results, map_solution, no_solution_results
from conin.util import try_import

from conin.representations.toulbar2.common import Contradiction

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401


def _solve_toulbar2(repn, q):
    """Solve a MAP query on a ``"toulbar2"`` representation of an MN, BN or DBN.

    ``q`` is the query of :func:`~conin.inference.map_query` (``variables``,
    ``evidence``, ``timing`` and ``options``).  The options are ``verbose``
    (the toulbar2 verbosity, default ``-1``), ``write_uai_file`` (write the
    UAI file that toulbar2 reads to this path), ``prune_network`` and
    ``create_MN`` (``True`` raises ``RuntimeError``), and the options of
    :func:`solve_toulbar2_map_query_model`.  An HMM is solved as its unrolled
    DBN (:func:`~conin.inference.hmm_results.solve_unrolled_hmm`).

    Raises
    ------
    ImportError
        If pytoulbar2 is not installed.
    """
    if not pytoulbar2_available:
        raise ImportError("method='toulbar2' needs pytoulbar2, which is not installed")
    options = dict(q.options)
    verbose = options.pop("verbose", -1)
    write_uai_file = options.pop("write_uai_file", None)
    reject_pruning_options(options)
    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    try:
        model = repn.create_model(
            evidence=q.evidence,
            variables=full_query_variables(q.variables, repn.nodes),
            verbose=verbose,
            write_uai_file=write_uai_file,
            timing=q.timing,
        )
    except Contradiction:
        # toulbar2 found while the model was built (a constraint that the
        # evidence or the other constraints contradict) that it has no solution.
        return no_solution_results(solver_timer.toc(None))
    return solve_toulbar2_map_query_model(model, timing=q.timing, **options)


def solve_toulbar2_map_query_model(
    model,
    *,
    solution_with_fixed=False,
    solution_with_evidence=False,
    timing=False,
):
    """Solve a toulbar2 MAP query model.

    Parameters
    ----------
    model : pytoulbar2.CFN
        A model built by the ``create_model()`` method of a ``"toulbar2"``
        representation (it has the ``V``, ``states`` and ``V_evidence``
        attributes that map the CFN variables back to the model's nodes).
    solution_with_fixed, solution_with_evidence : bool, optional
        Include the evidence variables in the solution (the two names are
        synonyms).
    timing : bool, optional
        Print timing information.

    Returns
    -------
    munch.Munch
        ``solution`` (``states``, ``{node: state}``; ``solver_objective``,
        the CFN's primal bound, an integer cost; ``log_potential`` is
        ``None``), ``solutions`` (``[solution]``), ``termination_condition``
        (``"ok"``) and ``solvetime``; see :mod:`conin.inference.results`.
        If the CFN has no solution (the constraints or the evidence exclude
        every assignment), ``solution`` is ``None``, ``solutions`` is empty
        and ``termination_condition`` is ``"error: no feasible solutions"``,
        as for A*.
    """
    solution_with_evidence = solution_with_fixed or solution_with_evidence
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("CFN_map_query - START")

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    res = model.Solve()
    solvetime = solver_timer.toc(None)
    if res is None:
        # pytoulbar2 returns None when no solution exists; conin sets no time
        # limit, so this is never a search that stopped early.
        if timing:  # pragma:nocover
            timer.toc("Completed optimization")
        return no_solution_results(solvetime)
    solution, primal_bound, num_solutions = res
    var = {
        name: model.states[i][solution[i]]
        for name, i in model.V.items()
        if solution_with_evidence or name not in model.V_evidence
    }
    soln = map_solution(var, solver_objective=primal_bound)

    if timing:  # pragma:nocover
        timer.toc("Completed optimization")

    return map_results([soln], solvetime=solvetime)
