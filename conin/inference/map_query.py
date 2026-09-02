from . import OptimizationInference
from . import CFNInference
from . import AStarInference
from . import ViterbiInference
from . import VariableElimination

_query_functions = {
    "integer_program": OptimizationInference._map_query_IntegerProgram,
    "toulbar2": CFNInference._map_query_Toulbar2,
    "a_star": AStarInference._map_query_AStar,
    "viterbi": ViterbiInference._map_query_Viterbi,
    "variable_elimination": VariableElimination._map_query_VariableElimination,
}


def map_query(
    pgm,
    *,
    method,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    **options,
):
    """Compute a MAP assignment

    This function uses ovld multiple dispatch to automatically select the
    appropriate backend inference function based on the type of the input model.

    Parameters
    ----------
    pgm : Graphical model to solve.
        Compatible pgmpy models are converted to CONIN model objects if
        pgmpy is installed.
    method : string
        The string name of the inference method.
    variables : list, optional
        Variables included in the MAP query. Support for partial MAP queries
        depends on the selected backend helper.
    evidence : dict or list, optional
        Observed variable assignments. For static models, a dictionary of
        ``{variable: state}``. For dynamic models, either a dense list or a
        dictionary keyed by time index.
    show_progress : bool, optional
        Whether to request solver progress reporting when supported.
    timing : bool, optional
        If ``True``, include timing information in the returned result.
        Only used by static model backends.
    start : int, optional
        Initial time index for dynamic models (ignored by static models).
    stop : int, optional
        Final time index for dynamic models (ignored by static models).
    options : dict, optional
        Additional keyword arguments forwarded to the selected
        inference helper, such as solver options or output-file settings.

    Returns
    -------
    munch.Munch
        Result object produced by the selected inference helper. The
        returned object typically contains ``solution.states`` and, when
        available, ``solvetime``.

    Raises
    ------
    TypeError
        If ``pgm`` is not a supported model type.

    Examples
    --------
    Static model (Markov network):

    >>> import conin.markov_network.examples
    >>> from conin.inference import map_query
    >>> example = conin.markov_network.examples.ABC_conin()
    >>> results = map_query(example.pgm, method="integer_program", solver=glpk)

    Dynamic model (Hidden Markov model):

    >>> import conin.hidden_markov_model.tests.examples
    >>> from conin.inference import map_query
    >>> pgm = conin.hidden_markov_model.tests.examples.create_hmm1()
    >>> results = map_query(pgm, method="integer_program", evidence=["o0", "o1"], solver=glpk)
    """
    if method not in _query_functions:
        raise ValueError(
            f"Unsupported inference type: {method}. "
            f"Expected one of: integer_program, toulbar2, a_star, viterbi, "
            f"variable_elimination"
        )
    return _query_functions[method](
        pgm,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        start=start,
        stop=stop,
        options=options,
    )
