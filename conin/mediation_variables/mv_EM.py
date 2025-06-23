import numpy as np
import itertools


def compute_emitweights(obs, hmm):
    """
    Separately handles the computation of the
    """
    T = len(obs)
    K = len(hmm.states)
    # Compute emissions weights for easier access
    emit_weights = np.zeros((T, K))
    for t in range(T):
        emit_weights[t] = np.array([hmm.eprob[k, obs[t]] for k in hmm.states])

    return emit_weights


def arrayConvert(obs, hmm, cst, sat):
    """
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    """
    # Initialize and convert all quantities  to np.arrays
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
    T = len(obs)
    K = len(hmm.states)
    M = len(aux_space)

    state_ix = {s: i for i, s in enumerate(hmm.states)}
    aux_ix = {s: i for i, s in enumerate(aux_space)}

    # Compute the hmm parameters
    tmat = np.zeros((K, K))
    init_prob = np.zeros(K)

    for i in hmm.states:
        init_prob[state_ix[i]] = hmm.initprob[i]
        for j in hmm.states:
            tmat[state_ix[i], state_ix[j]] = hmm.tprob[i, j]

    hmm_params = [tmat, init_prob]

    # Compute the cst parameters
    ind = np.zeros((M, K, M))
    init_ind = np.zeros((M, K))
    final_ind = np.zeros(M)

    for r in aux_space:
        final_ind[aux_ix[r]] = cst.cst_fun(r, sat)
        for i in hmm.states:
            init_ind[aux_ix[r], state_ix[i]] = cst.init_fun(i, r)
            for s in aux_space:
                ind[aux_ix[r], state_ix[i], aux_ix[s]] = cst.update_fun(r, i, s)

    cst_params = [init_ind, final_ind, ind]

    return hmm_params, cst_params


def mv_BaumWelch(hmm_params, emit_weights, cst_params, debug=False):
    """
    Baum-Welch algorithm that computes the moments in the M-step and returns the optimal init,tmat.
    Optimiziation of emissions will be handled separately since it's disribution-dependent.
    Maybe can add functionality if it needs the posterior moments.

    IN
    hmm_params (list) = [tmat,init_prob]. list of np.arrays. note that the emit_weights need to be computed beforehand
        tmat: (K,K) init_prob: (K)

    emit_weights. np.array of shape (T,K). the emission weights for each state. if updating emissions, need to recompute at every step too.

    cst_params (list) = [init_ind, final_ind, ind]. list of np.arrays. init/final_ind are handling first aux/final constraint emissions. ind is update.
        init_ind: (M,K) final_ind: (K) ind:(M,K,M)

    OUT

    the updated tmat, init_prob
    """
    # Initialize and convert all quantities  to np.arrays
    tmat, init_prob = hmm_params
    init_ind, final_ind, ind = cst_params
    T = emit_weights.shape[0]
    K = emit_weights.shape[1]
    M = init_ind.shape[0]

    # Initialize first
    alpha = np.empty((T, K, M))
    beta = np.empty(alpha.shape)

    alpha[0] = np.einsum("i,i,ri -> ir", emit_weights[0], init_prob, init_ind)
    beta[-1] = 1

    # Compute the forward pass
    for t in range(1, T):
        if t == (T - 1):
            alpha[t] = np.einsum(
                "i,ji,ris,js,r->ir", emit_weights[t], tmat, ind, alpha[t - 1], final_ind
            )
        else:
            alpha[t] = np.einsum(
                "i,ji,ris,js->ir", emit_weights[t], tmat, ind, alpha[t - 1]
            )

    # Compute the backward pass
    for t in range(1, T):
        if t == 1:
            beta[T - 1 - t] = np.einsum(
                "js,j,ij,sjr,s->ir",
                beta[T - t],
                emit_weights[T - t],
                tmat,
                ind,
                final_ind,
            )
        else:
            beta[T - 1 - t] = np.einsum(
                "js,j,ij,sjr->ir", beta[T - t], emit_weights[T - t], tmat, ind
            )

    # Compute P(Y,C=c), probability of observing emissions AND the constraint in the specified truth configuration
    prob_data = np.einsum(
        "ir,ir->", alpha[0], beta[0]
    )  # doesn't matter which time index. all give same

    # Compute first/second moments in M step
    gamma = 1 / prob_data * np.einsum("tir,tir->ti", alpha, beta)
    xi = (
        1
        / prob_data
        * np.einsum(
            "tjr,tk,jk,skr,tks->tjk",
            alpha[: (T - 1)],
            emit_weights[1:],
            tmat,
            ind,
            beta[1:],
        )
    )

    # Compute the optimal estimates
    pi_opt = gamma[0] / gamma[0].sum()
    tmat_opt = xi.sum(axis=0) / xi.sum(axis=(0, 2))[:, np.newaxis]

    if debug:
        prob_data = np.einsum("nir,nir -> n", alpha, beta)

    return [tmat_opt, pi_opt], prob_data


def mv_EM(
    obs, hmm, cst, sat=True, conv_tol=1e-8, max_iter=1000, emit_opt=None, debug=False
):

    # Convert everything into numpy arrays
    old_hmm_params, old_cst_params = arrayConvert(obs, hmm, cst, sat)
    emit_weights = compute_emitweights(obs, hmm)
    conv = 999
    it = 0
    while (conv > conv_tol) and (it <= max_iter):
        it += 1
        new_hmm_params, dat_prob = mv_BaumWelch(
            old_hmm_params, emit_weights, old_cst_params, debug=debug
        )
        # if emit_opt:
        #     emit_opt(*args) #args to be passed in and defined later.
        conv = np.linalg.norm(
            new_hmm_params[0] - old_hmm_params[0]
        )  # stopping criterion based on just transition matrix
        old_hmm_params = new_hmm_params

    return new_hmm_params, dat_prob


# def mv_BaumWelch(obs, hmm, cst, sat = True, emit_opt = None):
#     '''
#     Baum-Welch algorithm that computes the moments in the M-step and returns the optimal init,tmat.
#     If emissions also need to be optimized, then need to pass a optimizing function to emit_opt
#     '''
#     #Initialize and convert all quantities  to np.arrays
#     aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
#     T = len(obs)
#     K = len(hmm.states)
#     M = len(aux_space)
#     sat = True

#     state_ix = {s: i for i, s in enumerate(hmm.states)}
#     aux_ix = {s: i for i, s in enumerate(aux_space)}

#     tmat = np.zeros((K,K))
#     initprob_vec = np.zeros(K)

#     for i in hmm.states:
#         initprob_vec[state_ix[i]] = hmm.initprob[i]
#         for j in hmm.states:
#             tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]

#     ind = np.zeros((M,K,M))
#     init_ind = np.zeros((M,K))
#     final_ind = np.zeros(M)

#     for r in aux_space:
#         final_ind[aux_ix[r]] = cst.cst_fun(r,sat)
#         for i in hmm.states:
#             init_ind[aux_ix[r],state_ix[i]] = cst.init_fun(i,r)
#             for s in aux_space:
#                 ind[aux_ix[r],state_ix[i],aux_ix[s]] = cst.update_fun(r,i,s)

#     #Compute emissions weights for easier access
#     emit_weights = np.zeros((T,K))
#     for t in range(T):
#         emit_weights[t] = np.array([hmm.eprob[k,obs[t]] for k in hmm.states])

#     #Initialize first
#     alpha = np.empty((T,K,M))
#     beta = np.empty(alpha.shape)

#     curr_emits = np.array([hmm.eprob[k,obs[1]] for k in hmm.states])
#     alpha[0] = np.einsum('i,i,ri -> ir',curr_emits, initprob_vec,init_ind)
#     beta[-1] = 1

#     #Compute the forward pass
#     for t in range(1,T):
#         if t == (T-1):
#             alpha[t] = np.einsum('i,ji,ris,js,r->ir', emit_weights[t], tmat, ind, alpha[t-1], final_ind)
#         else:
#             alpha[t] = np.einsum('i,ji,ris,js->ir', emit_weights[t], tmat, ind, alpha[t-1])

#     #Compute the backward pass
#     for t in range(1,T):
#         if t == 1:
#             beta[T-1-t] = np.einsum('js,j,ij,sjr,s->ir', beta[T-t],emit_weights[T-t],tmat,ind, final_ind)
#         else:
#             beta[T-1-t] = np.einsum('js,j,ij,sjr->ir', beta[T-t],emit_weights[T-t],tmat,ind)

#     #Compute P(Y,C=c), probability of observing emissions AND the constraint in the specified truth configuration
#     prob_data  = np.einsum('ir,ir->',alpha[0],beta[0]) #doesn't matter which time index. all give same

#     #Compute first/second moments in M step
#     gamma = 1/prob_data*np.einsum('tir,tir->ti',alpha,beta)
#     xi = 1/prob_data*np.einsum('tjr,tk,jk,skr,tks->tjk',alpha[:(T-1)],emit_weights[1:],tmat,ind,beta[1:])

#     #Compute the optimal estimates
#     pi_opt = gamma[0]/gamma[0].sum()
#     tmat_opt = xi.sum(axis = 0)/xi.sum(axis = (0,2))[:,np.newaxis]

#     return pi_opt,tmat_opt,prob_data
