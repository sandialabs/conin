from conin.constrained_models import ConstrainedGraphicalModel
from conin.graphical_models import HiddenMarkovModel
from conin.graphical_models.assignments import as_sequence
from conin.algorithms.inference.hmm_a_star import hmm_a_star
from conin.algorithms.inference.results import (
    map_results,
    map_solution,
    score_hmm_solutions,
)


def a_star_(
    *,
    observed,
    oracle=None,
    hmm=None,
    num_solutions=1,
    max_iterations=None,
    max_time=None,
    max_prefixes=None,
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
        :mod:`conin.algorithms.inference.results`); the ``states`` of the solutions are
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
        max_prefixes=max_prefixes,
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
        Observed labels: a non-empty list or tuple (or a dict keyed by
        ``0..T-1``); anything else raises ``InvalidInputError``.
    hmm : HiddenMarkovModel or ConstrainedGraphicalModel (of an HMM)
        The model. Constraints must have an ``"oracle"`` representation (no
        constraints or ``"oracle"`` constraints); others raise
        ``InvalidInputError``.
    **kwargs
        ``num_solutions``, ``max_iterations``, ``max_time``, ``max_prefixes``
        and ``debug``.  ``max_iterations`` limits the number of prefixes
        expanded, ``max_time`` the time in seconds, and ``max_prefixes`` the
        number of prefixes kept in the open set, which bounds the memory the
        search uses; ``termination_condition`` names the limit that stopped
        the search (e.g. ``"max_prefixes: 1000001"``).

    Returns
    -------
    munch.Munch
        ``solution``, ``solutions``, ``termination_condition`` and
        ``solvetime``; each solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``) and
        ``solver_objective`` (``None``).  See :mod:`conin.algorithms.inference.results`.
        If no feasible sequence is found (the constraints exclude every
        sequence, or ``max_iterations``, ``max_time`` or ``max_prefixes``
        stops the search first),
        ``solution`` is ``None`` and ``solutions`` holds what was found.
    """
    cmodel = ConstrainedGraphicalModel.ensure(hmm, HiddenMarkovModel)
    # Raises InvalidInputError for constraint families without an "oracle"
    # representation (anything other than no constraints/"oracle").
    return _a_star(
        cmodel.representation("oracle"),
        observed=as_sequence(observed, "observed sequence"),
        **kwargs,
    )


def _a_star(
    oracle,
    *,
    observed,
    num_solutions=1,
    max_iterations=None,
    max_time=None,
    max_prefixes=None,
    debug=False,
):
    """:func:`a_star` on an ``"oracle"`` representation (external labels)."""
    matvec = oracle.hmm
    hmm = oracle.model
    if not oracle.constraints:
        oracle = None

    observed_ = [matvec.observable_to_internal[o] for o in observed]
    ans_ = a_star_(
        observed=observed_,
        oracle=oracle,
        hmm=matvec,
        num_solutions=num_solutions,
        max_iterations=max_iterations,
        max_time=max_time,
        max_prefixes=max_prefixes,
        debug=debug,
    )

    # Convert internal indices back to external labels
    solutions = [
        map_solution([matvec.hidden_states[h] for h in sol.states])
        for sol in ans_.solutions
    ]
    results = map_results(
        solutions,
        solvetime=ans_.solvetime,
        termination_condition=ans_.termination_condition,
    )
    return score_hmm_solutions(results, hmm, observed)
