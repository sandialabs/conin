"""toulbar2 MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models.hmm_observations import as_sequence, time_range
from conin.inference.hmm_results import solve_unrolled_hmm
from conin.inference.query_checks import map_query_args

from .solve import _solve_toulbar2


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
    observed = as_sequence(evidence)
    start, stop = time_range(observed, start, stop)
    timing = options.pop("timing", False)
    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    return solve_unrolled_hmm(
        _solve_toulbar2,
        cmodel.representation("toulbar2", start=start, stop=stop),
        map_query_args(
            variables=variables,
            evidence=evidence,
            observed=observed,
            timing=timing,
            options=dict(options, write_uai_file=write_uai_file),
        ),
    )
