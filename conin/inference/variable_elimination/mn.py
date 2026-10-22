"""Variable elimination MAP inference for (constrained) Markov networks."""

from conin.constrained_models import ConstrainedDiscreteMarkovNetwork

from .solve import _solve_static


def _map_query_variable_elimination_mn(
    repn,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
    solution_with_evidence=False,
):
    """Solve a MAP query on a ``"pgmpy"`` representation with variable elimination."""
    return _solve_static(
        repn,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        write_uai_file=write_uai_file,
        solution_with_evidence=solution_with_evidence,
        determine_variables=True,
    )


def inference_variable_elimination_map_query_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
    solution_with_evidence=False,
    **options,
):
    """MAP query on a Markov network with pgmpy's variable elimination.

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork
        Constraints must be of family ``"factor_oracle"`` or ``"algebraic"``.
    variables : list, optional
        The query variables (default: every node not in ``evidence``).
    evidence : dict, optional
        ``{node: state}``.
    show_progress : bool, optional
        Passed to pgmpy.
    timing : bool, optional
        Print timing information.
    write_uai_file : str, optional
        Save the factor model (with the constraint factors) to this file.
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
    cmodel = ConstrainedDiscreteMarkovNetwork.ensure(pgm)
    return _map_query_variable_elimination_mn(
        cmodel.representation("pgmpy"),
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        write_uai_file=write_uai_file,
        solution_with_evidence=solution_with_evidence,
    )
