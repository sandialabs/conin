import numpy as np


def hmm2numpy(hmm, ix_list=None, return_ix=False):
    """
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    """
    # Initialize and convert all quantities  to np.arrays

    if ix_list:
        state_ix, emit_ix = ix_list
    else:
        state_ix = {s: i for i, s in enumerate(hmm.states)}
        emit_ix = {s: i for i, s in enumerate(hmm.emits)}

    K = len(state_ix)
    M = len(emit_ix)
    # Compute the hmm parameters
    tmat = np.zeros((K, K))
    init_prob = np.zeros(K)

    emat = np.zeros((K, M))

    # Initial distribution.
    for i in hmm.states:
        if i not in hmm.initprob:
            continue
        init_prob[state_ix[i]] = hmm.initprob[i]

    # Transition matrix
    for i in hmm.states:
        for j in hmm.states:
            if (i, j) not in hmm.tprob:
                continue
            tmat[state_ix[i], state_ix[j]] = hmm.tprob[i, j]

    # Emission matrix
    for i in hmm.states:
        for m in hmm.emits:
            if (i, m) not in hmm.eprob:
                continue
            emat[state_ix[i], emit_ix[m]] = hmm.eprob[i, m]

    hmm_params = [init_prob, tmat, emat]

    if return_ix:
        return hmm_params, [state_ix, emit_ix]
    return hmm_params


def random_draw(p):
    """
    p is a 1D np array.
    single random draw from probability vector p and encode as 1-hot.
    """
    n = len(p)
    p = p / p.sum()
    draw = np.random.choice(n, p=p)
    one_hot = np.zeros(n, dtype=int)
    one_hot[draw] = 1

    return one_hot


def sample_hmm(hmm, time, constraint_checker=None, max_tries=1000):
    """
    INPUT:
        hmm: munch object
        time: int. time horizon.
        constraint_checker. Boolean function that ingests the generated hidden sequence and outputs whether it's feasible or not.
        max_tries: int. if constraint_checker, then how many tries to sample a feasible path.
    """
    # Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, return_ix=True)
    init_prob, tmat, emat = hmm_params

    # Prepare dictionary for converting one_hot back to states
    state_ix, emit_ix = ix_list
    state_ix = {v: k for k, v in state_ix.items()}
    emit_ix = {v: k for k, v in emit_ix.items()}

    num_tries = 0

    while num_tries < max_tries:
        # Generate (X1,Y1)
        x_prev = random_draw(init_prob)
        x_list = [state_ix[np.argmax(x_prev)]]  # convert one-hot back to state
        y_curr = random_draw(x_prev @ emat)
        y_list = [emit_ix[np.argmax(y_curr)]]

        # Generate rest
        for t in range(1, time):
            x_curr = random_draw(x_prev @ tmat)
            y_curr = random_draw(x_curr @ emat)
            x_list.append(state_ix[np.argmax(x_curr)])
            y_list.append(emit_ix[np.argmax(y_curr)])
            x_prev = x_curr

        if constraint_checker is not None:
            satisfied = constraint_checker(x_list)
            if satisfied:
                return x_list, y_list
        else:
            return x_list, y_list

    raise RuntimeError(f"Failed to sample a valid sequence after {max_tries} tries. ")
