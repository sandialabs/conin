"""toulbar2 MAP inference for (constrained) HMMs."""

from conin.constrained_models import ConstrainedHiddenMarkovModel

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
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
        The time steps (``stop`` defaults to ``len(evidence) - 1``).
    variables : list, optional
        Passed to the constraint functions as ``data.variables``.
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
    """
    if not pytoulbar2_available:
        return _not_available()

    if stop is None and evidence is not None:
        stop = len(evidence) - 1

    cmodel = ConstrainedHiddenMarkovModel.ensure(pgm)
    repn = cmodel.representation("toulbar2", start=start, stop=stop)
    if type(evidence) is list:
        dbn_evidence = {("E", i): evidence[i] for i in range(start, stop + 1)}
    else:
        dbn_evidence = {("E", k): v for k, v in evidence.items()}
    model = _create_model(
        repn,
        evidence=dbn_evidence,
        variables=variables,
        timing=options.get("timing", False),
        write_uai_file=write_uai_file,
        options=options,
    )
    results = solve_toulbar2_map_query_model(model, **options)
    if type(evidence) is list:
        results.solution.states = [
            results.solution.states["H", i] for i in range(start, stop + 1)
        ]
    else:
        results.solution.states = {
            i: results.solution.states["H", i] for i in range(start, stop + 1)
        }
    return results
