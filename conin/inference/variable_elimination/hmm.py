"""Variable elimination MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models.hmm_observations import (
    as_sequence,
    time_range,
    unrolled_evidence,
)
from conin.inference.hmm_results import hidden_states_from_unrolled

from .solve import _solve_repn, _with_timing


def _map_query_variable_elimination_hmm(
    repn,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    write_uai_file=None,
):
    """Solve a MAP query on the ``"factor"`` representation of an HMM."""

    def _execute():
        results = _solve_repn(
            repn,
            variables=variables,
            evidence=unrolled_evidence(as_sequence(evidence), repn.start, repn.stop),
            show_progress=show_progress,
            timing=timing,
            write_uai_file=write_uai_file,
            determine_variables=False,
        )
        return hidden_states_from_unrolled(
            results,
            start=repn.start,
            stop=repn.stop,
            keyed_by_time=isinstance(evidence, dict),
        )

    return _with_timing(timing, _execute)


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
    start, stop = time_range(as_sequence(evidence), start, stop)
    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    return _map_query_variable_elimination_hmm(
        cmodel.representation("factor", start=start, stop=stop),
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        write_uai_file=write_uai_file,
    )
