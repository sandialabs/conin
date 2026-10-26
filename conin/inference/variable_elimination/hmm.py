"""Variable elimination MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models.hmm_observations import as_sequence, time_range
from conin.inference.hmm_results import solve_unrolled_hmm
from conin.inference.query_checks import map_query_args

from .solve import _solve_variable_elimination


def inference_variable_elimination_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=None,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
    **options,
):
    """MAP query on an HMM (unrolled as a DBN) with pgmpy's variable elimination.

    Parameters
    ----------
    pgm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        Constraints must be of family ``"factor_oracle"`` or ``"algebraic"``;
        they see ``data = Munch(hmm=Munch(T))`` with
        ``T = list(range(start, stop + 1))``.
    start, stop : int, optional
        The time steps (``stop`` defaults to ``len(evidence) - 1``, the last
        observation; a later ``stop`` raises ``InvalidInputError``).
    variables : list, optional
        Nodes of the unrolled BN to query (default: every node not in the
        evidence).
    evidence : list, tuple or dict
        The observations: a list or tuple, or a dict keyed by the times
        ``0..T-1``.
    show_progress : bool, optional
        Passed to pgmpy.
    timing : bool, optional
        Print timing information.
    write_uai_file : str, optional
        Save the unrolled BN (with the constraint CPDs) to this file.
    **options
        Ignored.

    Returns
    -------
    munch.Munch
        ``solution.states`` is a list of hidden states (list evidence) or a
        dict keyed by time step (dict evidence).

    Raises
    ------
    InvalidInputError
        For evidence that is not a list, tuple or dict keyed by ``0..T-1``,
        or a ``stop`` past the last observation.
    """
    observed = as_sequence(evidence)
    start, stop = time_range(observed, start, stop)
    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    return solve_unrolled_hmm(
        _solve_variable_elimination,
        cmodel.representation("pgmpy", start=start, stop=stop),
        map_query_args(
            variables=variables,
            evidence=evidence,
            observed=observed,
            show_progress=show_progress,
            timing=timing,
            options=dict(write_uai_file=write_uai_file),
        ),
    )
