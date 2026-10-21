"""toulbar2 MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models.hmm_observations import (
    as_sequence,
    time_range,
    unrolled_evidence,
)
from conin.inference.hmm_results import hidden_states_from_unrolled
from conin.inference.query_checks import full_query_variables, reject_pruning_options

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
)


def _map_query_toulbar2_hmm(repn, *, variables, evidence, write_uai_file, options):
    """Solve a MAP query on the ``"toulbar2"`` representation of an HMM."""
    if not pytoulbar2_available:
        return _not_available()
    start, stop = repn.start, repn.stop
    dbn_evidence = unrolled_evidence(as_sequence(evidence), start, stop)
    reject_pruning_options(options)
    variables = full_query_variables(variables, repn.nodes)
    model = _create_model(
        repn,
        evidence=dbn_evidence,
        variables=variables,
        timing=options.get("timing", False),
        write_uai_file=write_uai_file,
        options=options,
    )
    results = solve_toulbar2_map_query_model(model, **options)
    return hidden_states_from_unrolled(
        results, start=start, stop=stop, keyed_by_time=isinstance(evidence, dict)
    )


def inference_toulbar2_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=None,
    variables=None,
    evidence=None,
    write_uai_file=None,
    **options,
):
    """MAP query on an HMM (unrolled as a DBN) with toulbar2.

    Parameters
    ----------
    pgm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
    start, stop : int, optional
        The time steps (``stop`` defaults to ``len(evidence) - 1``, the last
        observation; a later ``stop`` raises ``InvalidInputError``).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported (``RuntimeError``).
    evidence : list, tuple or dict
        The observations: a list or tuple, or a dict keyed by the times
        ``0..T-1``.
    write_uai_file : str, optional
        Write the UAI file that toulbar2 reads to this path.
    **options
        As for :func:`~conin.inference.toulbar2.inference_toulbar2_map_query_DDBN`.

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
    if not pytoulbar2_available:
        return _not_available()

    start, stop = time_range(as_sequence(evidence), start, stop)

    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    return _map_query_toulbar2_hmm(
        cmodel.representation("toulbar2", start=start, stop=stop),
        variables=variables,
        evidence=evidence,
        write_uai_file=write_uai_file,
        options=options,
    )
