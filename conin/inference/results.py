"""The shape of MAP query results.

Every MAP query (:func:`~conin.inference.map_query` and the search and solve
functions it dispatches to) returns a ``Munch`` with

``solution``
    The best solution, the first of ``solutions`` (``None`` if none was
    found).
``solutions``
    The solutions found, best first.
``termination_condition``
    ``"ok"``, or why the search stopped (for example
    ``"max_iterations: 10"`` or ``"error: no feasible solutions"``).
``solvetime``
    The time spent in the solver or the search, in seconds.

Some methods add a diagnostic key (``cut_stats`` for the integer program with
cut generation).  Each solution has

``states``
    The MAP assignment: ``{node: state}``, or for an HMM the hidden states (a
    list, or a dict keyed by time step if the observations were a dict).
``log_potential``
    The score of the assignment together with the evidence:
    :func:`conin.scoring.log_potential` of the model without its constraints.
    For a Bayesian network, DBN or HMM it is ``log P(x, e)`` (for an HMM,
    ``log P(hidden, observed)``), for a Markov network the sum of the log
    factors, ``log P(x, e) + log Z``.  It is not renormalised by the
    probability of the constraints, so it is comparable across methods.
    ``None`` for a partial MAP query, and from the functions that solve a
    backend model directly (``solve_*_map_query_model``), which do not see
    the model.
``solver_objective``
    The optimal objective value of the integer program or the toulbar2 cost
    function network, in the backend's own scale (not comparable across
    methods); ``None`` for the other methods.

The network-flow integer program adds ``variables`` (the nonzero model
variables) to each solution, and the MVR Viterbi adds ``augmented`` (the
decoded augmented path).
"""

import munch

from conin.graphical_models.hmm_observations import as_sequence
from conin.scoring.log_potential import hmm_path_log_potential, log_potential


def map_solution(states, *, solver_objective=None, **extra):
    """One solution of a MAP query (``log_potential`` is set by the scoring)."""
    return munch.Munch(
        states=states, log_potential=None, solver_objective=solver_objective, **extra
    )


def map_results(solutions, *, solvetime, termination_condition="ok", **extra):
    """The result of a MAP query; ``solution`` is the first of ``solutions``."""
    solutions = list(solutions)
    return munch.Munch(
        solution=solutions[0] if solutions else None,
        solutions=solutions,
        termination_condition=termination_condition,
        solvetime=solvetime,
        **extra,
    )


def score_solutions(results, model, *, evidence, nodes):
    """Set ``log_potential`` of the solutions of an MN, BN or DBN query.

    The assignment is the solution's ``states`` plus ``evidence``, restricted
    to ``nodes`` (the nodes of the model, unrolled for a DBN).  A solution
    that does not cover ``nodes`` (a partial MAP query) gets ``None``.
    """
    for soln in results.solutions:
        assignment = dict(evidence or {})
        assignment.update(soln.states)
        if all(node in assignment for node in nodes):
            soln.log_potential = log_potential(
                model, {node: assignment[node] for node in nodes}
            )
        else:
            soln.log_potential = None
    return results


def score_hmm_solutions(results, hmm, observed):
    """Set ``log_potential`` of the solutions of an HMM query.

    ``observed`` is the list of observations (the solutions cover the times
    ``0..len(states) - 1``), or a ``{time: label}`` dict of the observed
    times (the sparse observations of the MVR Viterbi, where an unobserved
    time contributes its transition only).
    """
    for soln in results.solutions:
        hidden = as_sequence(soln.states, "hidden states")
        if isinstance(observed, dict):
            obs = observed
        else:
            obs = dict(enumerate(observed[: len(hidden)]))
        soln.log_potential = hmm_path_log_potential(hmm, hidden, obs)
    return results
