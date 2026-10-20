"""Variable elimination MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel

from .solve import _solve_repn, _with_timing


def _hmm_evidence_to_dbn(evidence):
    """HMM observations (list or dict keyed by time step) -> ``{("E", t): obs}``."""
    if isinstance(evidence, list):
        return {("E", i): v for i, v in enumerate(evidence)}
    if isinstance(evidence, dict):
        return {("E", i): v for i, v in evidence.items()}
    return evidence


def _hmm_states_from_map(map_states, evidence):
    """Project the MAP assignment of the unrolled BN to the hidden states."""
    if isinstance(evidence, list):
        return [map_states["H", i] for i in range(len(map_states))]
    if isinstance(evidence, dict):
        return {i: map_states["H", i] for i in range(len(map_states))}
    return map_states


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
        The time steps (``stop`` defaults to ``len(evidence) - 1``).
    variables : list, optional
        Nodes of the unrolled BN to query (default: every node not in the
        evidence).
    evidence : list or dict
        The observations: a list, or a dict keyed by time step.
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
    """
    if stop is None:
        stop = len(evidence) - 1
    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)

    def _execute():
        repn = cmodel.representation("factor", start=start, stop=stop)
        results = _solve_repn(
            repn,
            variables=variables,
            evidence=_hmm_evidence_to_dbn(evidence),
            show_progress=show_progress,
            timing=timing,
            write_uai_file=write_uai_file,
            determine_variables=False,
        )
        results.solution.states = _hmm_states_from_map(
            results.solution.states, evidence
        )
        return results

    return _with_timing(timing, _execute)
