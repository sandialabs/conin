from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.hmm_a_star import hmm_a_star
from conin.inference.results import map_results, map_solution, score_hmm_solutions
from conin.representations.matvec import MatVecHMMRepresentation


def recursive_a_star(
    *,
    hmm,
    observed,
    num_solutions=1,
    debug=False,
    max_iterations=None,
    max_time=None,
):
    """A* search for the most likely hidden states under recursive constraints.

    The same search as :func:`conin.inference.oracle.a_star`, but prefixes
    with the same ``(last hidden state, length, constraint state)`` form one
    search node, which is expanded at most ``num_solutions`` times. The
    solutions are still the ``num_solutions`` most likely feasible sequences.

    Parameters
    ----------
    hmm : ConstrainedHiddenMarkovModel or HiddenMarkovModel
        The model.  Its constraints must be :class:`~conin.constraints.RecursiveConstraint`
        objects (or none); other families raise ``InvalidInputError``.
    observed : list
        Observed labels.
    num_solutions : int, optional
        Number of solutions to find.
    debug : bool, optional
        Print progress every 100 iterations.
    max_iterations : int, optional
        Maximum number of iterations.
    max_time : float, optional
        Maximum time in seconds.

    Returns
    -------
    munch.Munch
        ``solution`` (``None`` if no solution was found), ``solutions``,
        ``termination_condition`` and ``solvetime``; each solution has
        ``states`` (the hidden states), ``log_potential``
        (``log P(hidden, observed)``) and ``solver_objective`` (``None``).
        See :mod:`conin.inference.results`.
    """
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    return _recursive_a_star(
        cmodel.representation("recursive"),
        observed=observed,
        num_solutions=num_solutions,
        debug=debug,
        max_iterations=max_iterations,
        max_time=max_time,
    )


def _recursive_a_star(
    constraint,
    *,
    observed,
    num_solutions=1,
    debug=False,
    max_iterations=None,
    max_time=None,
):
    """:func:`recursive_a_star` on a ``"recursive"`` representation."""
    hmm = constraint.model
    h2e = hmm.hidden_to_external
    observed_ = [hmm.observed_to_internal[o] for o in observed]
    ans_ = hmm_a_star(
        MatVecHMMRepresentation(hmm),
        observed_,
        initial=lambda h: constraint.initial(h2e[h]),
        update=lambda state, h: constraint.update(state, h2e[h]),
        feasible_partial=lambda state, seq, T: constraint.feasible_partial(
            state, len(seq), T
        ),
        feasible=lambda state, seq: constraint.feasible(state),
        merge=True,
        num_solutions=num_solutions,
        max_iterations=max_iterations,
        max_time=max_time,
        debug=debug,
    )

    # Convert internal indices back to external labels
    solutions = [map_solution([h2e[h] for h in sol.states]) for sol in ans_.solutions]
    results = map_results(
        solutions,
        solvetime=ans_.solvetime,
        termination_condition=ans_.termination_condition,
    )
    return score_hmm_solutions(results, hmm, observed)
