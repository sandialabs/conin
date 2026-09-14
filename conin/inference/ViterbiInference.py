from ovld import ovld
from conin.hidden_markov_model.hmm import HiddenMarkovModel, HMM_MatVecRepn
from conin.hidden_markov_model.inference import viterbi


@ovld
def _map_query_Viterbi(
    pgm,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    """Compute a MAP assignment using Viterbi algorithm with multiple dispatch."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: HiddenMarkovModel, HMM_MatVecRepn."
    )


def _run_viterbi(pgm, variables, evidence):
    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = viterbi(observed=observed, hmm=pgm)
        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results
    elif type(evidence) is list:
        return viterbi(observed=evidence, hmm=pgm)


@ovld
def _map_query_Viterbi(
    pgm: HiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return _run_viterbi(pgm, variables, evidence)


@ovld
def _map_query_Viterbi(
    pgm: HMM_MatVecRepn,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return _run_viterbi(pgm, variables, evidence)
