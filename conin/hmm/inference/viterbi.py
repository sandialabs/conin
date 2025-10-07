import numpy as np
import munch

from conin.hmm import HiddenMarkovModel, HMM


def viterbi_(*, observed, hmm):
    """
    Performs the Viterbi algorithm to find the most likely sequence of hidden states.

    This function performs inference using an instance of the HMM class, which defines
    a matrix/vector representation of a hidden Markov model.

    Parameters:
        observed (list): The sequence of observations to perform inference on.
        hmm (HMM): The HMM model to use for inference.

    Returns:
        list: The most likely sequence of hidden states.
    """
    with np.errstate(divide="ignore"):  # TODO Is this the best way to deal with this?
        # Initalize variables
        time_steps = len(observed)
        viterbi_recursion = np.zeros((hmm.num_hidden_states, time_steps))
        backpointer = np.zeros((hmm.num_hidden_states, time_steps), dtype=int)

        log_eprob = {
            (h1, o): np.log(hmm.emission_mat[h1][o])
            for h1 in hmm.hidden_states
            for o in hmm.observed_states
        }
        log_tprob = {
            (h1, h2): np.log(hmm.transition_mat[h1][h2])
            for h1 in hmm.hidden_states
            for h2 in hmm.hidden_states
        }

        # Initialization step
        for h in hmm.hidden_states:
            obs = observed[0]
            viterbi_recursion[h, 0] = np.log(hmm.start_vec[h]) + log_eprob[h, obs]
            backpointer[h, 0] = -1

        # Recursion step
        for t in range(1, time_steps):
            obs = observed[t]
            for h1 in hmm.hidden_states:
                max_prob = -np.inf
                max_state = -1
                e_prob = log_eprob[h1, obs]
                for h2 in hmm.hidden_states:
                    prob = viterbi_recursion[h2, t - 1] + log_tprob[h2, h1] + e_prob
                    if prob > max_prob:
                        max_prob = prob
                        max_state = h2
                viterbi_recursion[h1, t] = max_prob
                backpointer[h1, t] = max_state

        # Termination step
        max_prob = -np.inf
        last_state = -1
        for h in range(hmm.num_hidden_states):
            if viterbi_recursion[h, time_steps - 1] > max_prob:
                max_prob = viterbi_recursion[h, time_steps - 1]
                last_state = h

        # Path backtracking
        hidden = [0] * time_steps
        hidden[time_steps - 1] = last_state
        for t in range(time_steps - 2, -1, -1):
            hidden[t] = backpointer[hidden[t + 1], t + 1]

        soln = munch.Munch(
            variable_value=hidden, hidden=hidden, log_likelihood=max_prob
        )
        ans = munch.Munch(
            observations=observed,
            solution=soln,
            solutions=[soln],
            termination_condition="ok",
        )
        return ans


def viterbi(*, observed, hmm):
    """
    Performs the Viterbi algorithm to find the most likely sequence of hidden states.

    Parameters:
        observed (list): The sequence of observed to perform inference on.
        hmm (HiddenMarkovModel or HMM): The HMM to use for inference.

    Returns:
        list: The most likely sequence of hidden states.
    """
    if isinstance(hmm, HMM):
        return viterbi_(observed=observed, hmm=hmm)

    # ELSE isinstance(hmm, HiddenMarkovModel)

    observed_ = [hmm.observed_to_internal[o] for o in observed]
    hmm_ = hmm.hmm  # HMM instance associated with the HiddenMarkovModel
    ans_ = viterbi_(observed=observed_, hmm=hmm_)

    # Convert internal indices back to external labels
    solutions = []
    for sol in ans_.solutions:
        hidden = [hmm.hidden_to_external[h] for h in sol.hidden]
        solutions.append(
            munch.Munch(
                variable_value=hidden, hidden=hidden, log_likelihood=sol.log_likelihood
            )
        )

    return munch.Munch(
        observations=observed,
        solution=solutions[0],
        solutions=solutions,
        termination_condition=ans_.termination_condition,
    )
