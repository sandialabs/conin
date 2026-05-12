import munch
import conin.markov_network
from conin.hidden_markov_model import ConstrainedHiddenMarkovModel
from conin.hidden_markov_model.hmm_to_dbn import create_dbn_from_hmm
from conin.inference.dbn.inference_toulbar2 import (
    create_toulbar2_map_query_model_DDBN,
)


def create_toulbar2_map_query_model_HMM(
    *, pgm, start=0, stop=None, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.hidden_markov_model if isinstance(pgm, ConstrainedHiddenMarkovModel) else pgm
    )

    dbn = create_dbn_from_hmm(hmm=pgm_)
    if type(evidence) is list:
        evidence = {("E", i): evidence[i] for i in range(start, stop + 1)}
    else:
        evidence = {("E", k): v for k, v in evidence.items()}


    model = create_toulbar2_map_query_model_DDBN(
        pgm=dbn,
        variables=variables,
        evidence=evidence,
        start=start,
        stop=stop,
        **options,
    )

    if isinstance(pgm, ConstrainedHiddenMarkovModel) and pgm.constraints:
        data = munch.Munch(
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            T=list(range(start, stop + 1)),
        )
        for func in pgm.constraints:
            model = func(model, data)

    return model


def inference_toulbar2_map_query_HMM(
    *,
    pgm,
    start=0,
    stop=None,
    variables=None,
    evidence=None,
    **options,
):
    if stop is None and evidence is not None:
        stop = len(evidence)-1

    model = create_toulbar2_map_query_model_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )
    results = conin.inference.mn.inference_toulbar2.solve_toulbar2_map_query_model(
        model, **options
    )
    if type(evidence) is list:
        results.solution.states = [
            results.solution.states["H", i] for i in range(start, stop + 1)
        ]
    else:
        results.solution.states = {
            i: results.solution.states["H", i] for i in range(start, stop + 1)
        }
    return results

