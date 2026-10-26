"""Variable elimination MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork

from .solve import _solve_static


def _map_query_variable_elimination_dbn(
    repn,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
    solution_with_evidence=False,
):
    """Solve a MAP query on the ``"pgmpy"`` representation of a DBN."""
    return _solve_static(
        repn,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        write_uai_file=write_uai_file,
        solution_with_evidence=solution_with_evidence,
        determine_variables=False,
    )


def inference_variable_elimination_map_query_DDBN(
    *,
    pgm,
    start=0,
    stop=1,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
    solution_with_evidence=False,
    **options,
):
    """MAP query on a DBN unrolled over ``start..stop`` with pgmpy's variable elimination.

    Parameters
    ----------
    pgm : DynamicDiscreteBayesianNetwork or ConstrainedDynamicDiscreteBayesianNetwork
        Constraints must be of family ``"factor_oracle"`` or ``"algebraic"``;
        they see ``data = Munch(T)`` with ``T = list(range(start, stop + 1))``.
    start, stop : int, optional
        The time steps (``stop=None`` means ``1``).
    variables : list, optional
        The query variables (default: every node not in ``evidence``).
    evidence : dict, optional
        ``{(node, t): state}``.
    show_progress : bool, optional
        Passed to pgmpy.
    timing : bool, optional
        Print timing information.
    write_uai_file : str, optional
        Save the unrolled BN (with the constraint CPDs) to this file.
    solution_with_evidence : bool, optional
        Add ``evidence`` to ``solution.states``.  Always done when the model
        has constraints.
    **options
        Ignored.

    Returns
    -------
    munch.Munch
        ``solution.states`` and ``solvetime``.
    """
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    return _map_query_variable_elimination_dbn(
        cmodel.representation("pgmpy", start=start, stop=stop),
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        write_uai_file=write_uai_file,
        solution_with_evidence=solution_with_evidence,
    )
