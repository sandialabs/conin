from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.hmm_a_star import hmm_a_star
from conin.inference.results import map_results, map_solution, score_hmm_solutions


def a_star_(
    *,
    observed,
    oracle=None,
    hmm=None,
    num_solutions=1,
    max_iterations=None,
    max_time=None,
    debug=False,
):
    """A* search on internal indices, with optional oracle feasibility checks.

    Parameters:
        observed (list): The internal indices of the observed states
        oracle (OracleHMMRepresentation, optional): The feasibility checks
            (``representation("oracle")`` of a constrained HMM). A prefix is
            extended by a state only if ``oracle.partial_is_feasible`` accepts
            the extended prefix, and a full sequence is a solution only if
            ``oracle.is_feasible`` accepts it.
        hmm (MatVecHMMRepresentation): The HMM parameters
            (``representation("matvec")`` of the HMM; ``oracle.hmm`` if omitted)

    Returns:
        munch.Munch: ``solution`` (``None`` if no solution was found),
        ``solutions``, ``termination_condition`` and ``solvetime`` (see
        :mod:`conin.inference.results`); the ``states`` of the solutions are
        tuples of internal indices and their ``log_potential`` is the search
        score.
    """
    if oracle is not None and hmm is None:
        hmm = oracle.hmm

    if oracle is None:
        checks = {}
    else:
        # No search state: every prefix is its own node.
        checks = dict(
            feasible_partial=lambda state, seq, T: oracle.partial_is_feasible(
                T=T, seq=seq
            ),
            feasible=lambda state, seq: oracle.is_feasible(seq),
        )
    return hmm_a_star(
        hmm,
        observed,
        num_solutions=num_solutions,
        max_iterations=max_iterations,
        max_time=max_time,
        debug=debug,
        **checks,
    )


def a_star(
    *,
    observed,
    hmm,
    **kwargs,
):
    """A* search for the most likely hidden states of an (oracle-)constrained HMM.

    Parameters
    ----------
    observed : list
        Observed labels.
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model. Constraints must have an ``"oracle"`` representation (no
        constraints or ``"oracle"`` constraints); others raise
        ``InvalidInputError``.
    **kwargs
        ``num_solutions``, ``max_iterations``, ``max_time``, ``debug``.

    Returns
    -------
    munch.Munch
        ``solution``, ``solutions``, ``termination_condition`` and
        ``solvetime``; each solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``) and
        ``solver_objective`` (``None``).  See :mod:`conin.inference.results`.
        If no feasible sequence is found (the constraints exclude every
        sequence, or ``max_iterations``/``max_time`` stops the search first),
        ``solution`` is ``None`` and ``solutions`` holds what was found.
    """
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    # Raises InvalidInputError for constraint families without an "oracle"
    # representation (anything other than no constraints/"oracle").
    return _a_star(cmodel.representation("oracle"), observed=observed, **kwargs)


def _a_star(oracle, *, observed, **kwargs):
    """:func:`a_star` on an ``"oracle"`` representation (external labels)."""
    matvec = oracle.hmm
    hmm = oracle.model
    if not oracle.constraints:
        oracle = None

    observed_ = [hmm.observed_to_internal[o] for o in observed]
    ans_ = a_star_(observed=observed_, oracle=oracle, hmm=matvec, **kwargs)

    # Convert internal indices back to external labels
    solutions = [
        map_solution([hmm.hidden_to_external[h] for h in sol.states])
        for sol in ans_.solutions
    ]
    results = map_results(
        solutions,
        solvetime=ans_.solvetime,
        termination_condition=ans_.termination_condition,
    )
    return score_hmm_solutions(results, hmm, observed)
