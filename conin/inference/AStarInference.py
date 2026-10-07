from ovld import ovld
from conin.hidden_markov_model.hmm import HiddenMarkovModel, HMM_MatVecRepn
from conin.hidden_markov_model import ConstrainedHiddenMarkovModel, CHMM
from conin.hidden_markov_model.inference import a_star


@ovld
def _map_query_AStar(
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
    """Compute a MAP assignment using A* search with multiple dispatch."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: HiddenMarkovModel, HMM_MatVecRepn, "
        f"ConstrainedHiddenMarkovModel, CHMM."
    )


@ovld
def _map_query_AStar(
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
    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = a_star(observed=observed, hmm=pgm, **options)
        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results
    elif type(evidence) is list:
        return a_star(observed=evidence, hmm=pgm, **options)


@ovld
def _map_query_AStar(
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
    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = a_star(observed=observed, hmm=pgm, **options)
        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results
    elif type(evidence) is list:
        return a_star(observed=evidence, hmm=pgm, **options)


@ovld
def _map_query_AStar(
    pgm: ConstrainedHiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = a_star(observed=observed, hmm=pgm, **options)
        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results
    elif type(evidence) is list:
        return a_star(observed=evidence, hmm=pgm, **options)


@ovld
def _map_query_AStar(
    pgm: CHMM,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    if type(evidence) is dict:
        observed = [evidence[i] for i in range(len(evidence))]
        results = a_star(observed=observed, hmm=pgm, **options)
        solutions = results.solutions
        for soln in solutions:
            soln.states = {i: v for i, v in enumerate(soln.states)}
            soln.hidden = soln.states
        results.solutions = solutions
        return results
    elif type(evidence) is list:
        return a_star(observed=evidence, hmm=pgm, **options)
