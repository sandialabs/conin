"""toulbar2 MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference._hmm_evidence import (
    hmm_dbn_evidence,
    hmm_states_from_dbn,
    require_hmm_evidence,
)
from conin.inference._query_checks import validate_bn_query

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
    dbn_evidence = hmm_dbn_evidence(evidence, start, stop)
    validate_bn_query(options, variables, repn.nodes, evidence=dbn_evidence)
    model = _create_model(
        repn,
        evidence=dbn_evidence,
        variables=variables,
        timing=options.get("timing", False),
        write_uai_file=write_uai_file,
        options=options,
    )
    results = solve_toulbar2_map_query_model(model, **options)
    return hmm_states_from_dbn(results, evidence, start, stop)


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
        The time steps (``stop`` defaults to ``len(evidence) - 1``).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported (``RuntimeError``).
    evidence : list or dict
        The observations: a list, or a dict keyed by time step.
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
        For evidence that is not a list or dict.
    """
    if not pytoulbar2_available:
        return _not_available()

    require_hmm_evidence(evidence)
    if stop is None:
        stop = len(evidence) - 1

    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    return _map_query_toulbar2_hmm(
        cmodel.representation("toulbar2", start=start, stop=stop),
        variables=variables,
        evidence=evidence,
        write_uai_file=write_uai_file,
        options=options,
    )
