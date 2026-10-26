import numpy as np
from pyomo.common.timing import TicTocTimer

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.results import map_results, map_solution, score_hmm_solutions


def viterbi_(*, observed, hmm):
    """
    Performs the Viterbi algorithm to find the most likely sequence of hidden states.

    This function works on internal indices.

    Parameters:
        observed (list): The internal indices of the observed states.
        hmm (MatVecHMMRepresentation): The HMM parameters
            (``representation("matvec")`` of the HMM).

    Returns:
        munch.Munch: The result (see :mod:`conin.inference.results`); the
        ``states`` of the solution are internal indices and its
        ``log_potential`` is the Viterbi score.
    """
    timer = TicTocTimer()
    timer.tic(None)
    time_steps = len(observed)
    hidden_states = range(hmm.num_hidden_states)
    # Lists of log-probabilities (-inf for zeros): faster to index than arrays
    log_start = hmm.log_start_vec.tolist()
    log_tprob = hmm.log_transition_mat.tolist()
    log_eprob = hmm.log_emission_mat.tolist()
    viterbi_recursion = np.zeros((hmm.num_hidden_states, time_steps))
    backpointer = np.zeros((hmm.num_hidden_states, time_steps), dtype=int)

    # Initialization step
    obs = observed[0]
    for h in hidden_states:
        viterbi_recursion[h, 0] = log_start[h] + log_eprob[h][obs]
        backpointer[h, 0] = -1

    # Recursion step
    for t in range(1, time_steps):
        obs = observed[t]
        for h1 in hidden_states:
            max_prob = -np.inf
            max_state = -1
            e_prob = log_eprob[h1][obs]
            for h2 in hidden_states:
                prob = viterbi_recursion[h2, t - 1] + log_tprob[h2][h1] + e_prob
                if prob > max_prob:
                    max_prob = prob
                    max_state = h2
            viterbi_recursion[h1, t] = max_prob
            backpointer[h1, t] = max_state

    # Termination step
    max_prob = -np.inf
    last_state = -1
    for h in hidden_states:
        if viterbi_recursion[h, time_steps - 1] > max_prob:
            max_prob = viterbi_recursion[h, time_steps - 1]
            last_state = h

    # Path backtracking
    hidden = [0] * time_steps
    hidden[time_steps - 1] = last_state
    for t in range(time_steps - 2, -1, -1):
        hidden[t] = backpointer[hidden[t + 1], t + 1]

    soln = map_solution(hidden)
    soln.log_potential = max_prob
    return map_results([soln], solvetime=timer.toc(None))


def viterbi(*, observed, hmm):
    """
    Performs the Viterbi algorithm to find the most likely sequence of hidden states.

    Parameters:
        observed (list): The sequence of observed labels to perform inference on.
        hmm (HiddenMarkovModel or ConstrainedHiddenMarkovModel): The HMM to use
            for inference.  A constrained model must have no constraints
            (otherwise ``InvalidInputError`` is raised).

    Returns:
        munch.Munch: ``solution``, ``solutions``, ``termination_condition``
        and ``solvetime``; the solution has ``states`` (the hidden states),
        ``log_potential`` (``log P(hidden, observed)``) and
        ``solver_objective`` (``None``).  See :mod:`conin.inference.results`.
    """
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    # Raises InvalidInputError if there are constraints.
    return _viterbi(cmodel.representation("matvec"), observed=observed)


def _viterbi(repn, *, observed):
    """:func:`viterbi` on a ``"matvec"`` representation (external labels)."""
    hmm = repn.model
    observed_ = [hmm.observed_to_internal[o] for o in observed]
    ans_ = viterbi_(observed=observed_, hmm=repn)

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
