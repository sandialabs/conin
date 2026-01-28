import numpy as np
import torch
import json
from munch import Munch
import itertools
import copy
import time



######################################################################
# MVR
######################################################################

def create_cst_params(cst, hidden_states, cst_params = None, dtype = torch.float32, device = 'cpu'):
    m_states = cst.m_states
    if cst_params:
        cst_params = [c.to(device) for c in cst_params]
        init_mat, upd_mat, eval_mat = cst_params
    else:
        init = cst.init_fun
        upd = cst.update_fun
        eval_fun = cst.eval_fun

        #returns a (k,s,r) array. k is current hideen. r,s are present/past mediation.
        upd_mat = torch.tensor([[[upd(k,r,s) for s in m_states] for r in m_states] for k in hidden_states], dtype = dtype, device = device)

        #returns a (k,r) array. k,r are current hidden/mediation states
        init_mat = torch.tensor([[init(k,r) for r in m_states] for k in hidden_states], dtype = dtype, device = device)

        #return (k,r) array for terminal emission.
        eval_mat = torch.tensor([[eval_fun(k,r) for r in m_states] for k in hidden_states], dtype = dtype, device = device)

    return init_mat, eval_mat, upd_mat

def create_seqconstraint(K,M):
    '''
    Given state space [0,...,K-1] and constraint of encountering [0,...,N-1]
    '''
    if M > K:
        raise ValueError(f'subsequence length {M} cannot exceed number of states {K}')

    def update_fun(k , r, r_past):
        '''
        have a counter. since sequence is [0,...,N-1], the counter also encodes the current expected state in the sequence
        '''
        if r_past == M: #once we hit M, we stay there
            new_counter = M
        else:
            if k == r_past:
                new_counter = r_past + 1
            else:
                new_counter = int(k == 0)
        return r == (new_counter)
    def init_fun(k, r):
        '''
        initial "prob" of r = (m1,m2) from k. is just indicator
        '''
        return r == int(k == 0)
    def eval_fun(k, r):
        return r == M
    m_states = list(range(M+1))
    seq_cst = Munch(update_fun = update_fun, init_fun = init_fun, eval_fun = eval_fun, m_states = m_states)

    return seq_cst


def create_seq_cst_params(K, M, device='cpu', dtype=torch.float32):
    """
    Create all tensors needed for sequence tracking.
    
    Args:
        K: Size of symbol alphabet [0, ..., K-1]
        M: Length of sequence to track (M <= K). Desired sequence is: [0,...,M-1]
        device: Device to place tensors on (default: 'cpu')
    
    Returns:
        List of [init_mat, update_mat, eval_mat]
        - update_mat: (K, M+1, M+1) transition tensor
          Ordered as [current_symbol, current_state, previous_state]
        - init_mat: (K, M+1) initial state vector
        - eval_mat: (K, M+1) final/accepting state vector
    """
    assert M <= K, "M must be <= K"
    
    # State space is now just [0, 1, ..., M]
    # where state i means we've seen [0, ..., i-1] and are looking for i next
    
    # ===== UPDATE MATRIX =====
    # Tensor shape: (K, M+1, M+1)
    # Order: [current_symbol, current_state, previous_state]
    update_mat = torch.zeros(K, M + 1, M + 1, device=device, dtype=dtype)
    
    # Rule 0a: Staying in completed state
    # Once at state M (meaning have encountered sequence), stay there.
    update_mat[:, M, M] = 1
    
    # Rule 0b: Reset to zero when counter < M and current symbol > 0. Will undo since too broad. 
    for k in range(1, K):
        update_mat[k, 0, :M] = 1
        
    # Rule 1: Advancing through the sequence
    # When seeing symbol i where 1 < i < M and i == counter
    # Also undo Rule0b.
    for i in range(1,M):
        update_mat[i, i + 1, i] = 1
        update_mat[i, 0, i] = 0
        
    # Rule 2: Reset/restart on seeing 0
    # When seeing symbol 0, go to state 1 from any state < M
    update_mat[0, 1, :M] = 1
    
    # ===== INIT MATRIX =====
    # Tensor shape: (K, M+1)
    init_mat = torch.zeros(K, M + 1, device=device, dtype=dtype)
    
    # Symbol 0: start at state 1 (we've seen [0])
    init_mat[0, 1] = 1
    
    # Symbol i>0: start at state 0 (we've seen nothing, looking for 0)
    init_mat[1:, 0] = 1
    
    # ===== EVAL MATRIX =====
    # Tensor shape: (K, M+1)
    eval_mat = torch.zeros(K, M + 1, device=device, dtype=dtype)
    
    # Accept only when in state M (completed the sequence)
    eval_mat[:, M] = 1
    
    return [init_mat, update_mat, eval_mat]

def convertTensor_list(hmm, cst_list, dtype = torch.float16, device = 'cpu', hmm_params = None, cst_params = None, return_ix = False):
    '''
    cst_list is a list of the individual csts.
    '''
    #Initialize and convert all quantities  to np.arrays
    hmm = copy.deepcopy(hmm)
    K = len(hmm.states)
    
    state_ix = {s: i for i, s in enumerate(hmm.states)}
    
    #Compute the hmm parameters if not provided
    if hmm_params is None:
        tmat = torch.zeros((K,K), dtype=dtype ).to(device)
        init_prob = torch.zeros(K, dtype=dtype ).to(device)
    
        for i in hmm.states:
            init_prob[state_ix[i]] = hmm.initprob[i]
            for j in hmm.states:
                tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]
    
        hmm_params = [tmat, init_prob]
    
    else:
        hmm_params = [h.to(device) for h in hmm_params]
        
    
    #Compute the cst parameters 
    init_list = []
    eval_list = []
    upd_list = []
    dims_list = []
    cst_ix = 0
    C = len(cst_list)

    #indices are (hidden, c_1,....,c_C, hidden, c_1,....,c_C) are augmented messages
    for cst in cst_list:
        cst = copy.deepcopy(cst)
        init_mat, eval_mat, upd_mat = create_cst_params(cst, hmm.states, cst_params = cst_params, dtype = dtype, device = device)
        init_list += [init_mat,[0,cst_ix + 1]]
        eval_list += [eval_mat, [0, cst_ix + 1]]
        upd_list += [upd_mat, [0, cst_ix + 1,cst_ix + C + 2]]
        dims_list.append(len(cst.m_states))
        cst_ix += 1
                
    cst_params = [dims_list, init_list,eval_list,upd_list]

    if return_ix:
        return hmm_params, cst_params, state_ix
    return hmm_params, cst_params 

def compute_emitweights(obs,hmm):
    '''
    Separately handles the computation of the 
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    T = len(obs)
    K = len(hmm.states)
    #Compute emissions weights for easier access
    emit_weights = np.zeros((T,K))
    for t in range(T):
        emit_weights[t] = np.array([hmm.eprob[k,obs[t]] for k in hmm.states])
    return emit_weights

# def hmm2numpy(hmm, ix_list = None, return_ix = False):
#     '''
#     Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
#     By assumption, the update/emission parameters associated with the constraint are static.
#     For now, fix the emission probabilities.
#     Only the hmm paramters are being optimized.
#     '''
#     #Initialize and convert all quantities  to np.arrays

#     if ix_list:
#         state_ix, emit_ix = ix_list
#     else:
#         state_ix = {s: i for i, s in enumerate(hmm.states)}
#         emit_ix = {s: i for i, s in enumerate(hmm.emits)}

#     K = len(state_ix)
#     M = len(emit_ix)
#     #Compute the hmm parameters
#     tmat = np.zeros((K,K))
#     init_prob = np.zeros(K)

#     emat = np.zeros((K,M))

#     #Initial distribution. 
#     for i in hmm.states:
#         if i not in hmm.initprob:
#             continue
#         init_prob[state_ix[i]] = hmm.initprob[i]

#     #Transition matrix
#     for i in hmm.states:
#         for j in hmm.states:
#             if (i,j) not in hmm.tprob:
#                 continue
#             tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]

    
#     #Emission matrix
#     for i in hmm.states:
#         for m in hmm.emits:
#             if (i,m) not in hmm.eprob:
#                 continue
#             emat[state_ix[i],emit_ix[m]] = hmm.eprob[i,m]

#     hmm_params = [init_prob, tmat, emat]

#     if return_ix:
#         return hmm_params, [state_ix, emit_ix] 
#     return hmm_params

def Viterbi_preprocess(hmm, cst_list, obs, dtype = torch.float32,  device = 'cpu', debug = False, hmm_params = None, cst_params = None):
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, dtype = dtype, \
                                                               device = device, return_ix = True, hmm_params = hmm_params, cst_params = cst_params)   

    return [emit_weights, hmm_params, cst_params_list, state_ix]

def Viterbi_torch_list(input_list, obs, dtype = torch.float32,  device = 'cpu', debug = False, num_corr = 0, hmm_params = None):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.
    '''
    emit_weights, hmm_params, cst_params_list, state_ix = input_list
    tmat, init_prob = hmm_params
    dims_list, init_ind_list,final_ind_list,ind_list = cst_params_list

    
    #Viterbi
    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)
    
    val = torch.empty((T,K) + tuple(dims_list), device = 'cpu')
    ix_tracker = torch.empty((T,K) + tuple(dims_list), device = 'cpu') #will store flattened indices
    
    kr_indices = list(range(C+1))
    kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]
    
    
    #Forward pass
    # V = torch.einsum('k,k,kr -> kr', init_prob, emit_weights[0], init_ind)
    V = torch.einsum(emit_weights[0], [0], init_prob, [0], *init_ind_list, kr_indices)
    V = V/(V.max() + num_corr) #normalize for numerical stability
    val[0] = V.cpu()
    
    for t in range(1,T):
        # return kr_indices, ind_list, dims_list, C
        # V = torch.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = torch.einsum(val[t-1].to(device), js_indices, tmat, [C+1,0], *ind_list, list(range(2*C + 2)))
        V = V.reshape(tuple(kr_shape) + (-1,))
        V = V/(V.max() + num_corr)
        max_ix = torch.argmax(V, axis = -1, keepdims = True)
        ix_tracker[t-1] = max_ix.squeeze()
        V = torch.take_along_dim(V, max_ix, axis=-1).squeeze()

        if t == T -1:
            # Evaluate all constraints at the last time
            val[t] = torch.einsum(emit_weights[t], [0], V, kr_indices, *final_ind_list, kr_indices).cpu()
        else:
            # Regular update without evaluating constraints
            val[t] = torch.einsum(emit_weights[t], [0], V, kr_indices, kr_indices).cpu()


    # return val
    state_ix = {v:k for k,v in state_ix.items()}
    #Backward pass
    opt_augstateix_list = []
    max_ix = int(torch.argmax(val[T-1]).item())
    unravel_max_ix = np.unravel_index(max_ix, kr_shape)
    opt_augstateix_list =  [np.array(unravel_max_ix).tolist()] + opt_augstateix_list
    
    ix_tracker = ix_tracker.reshape(T,-1) #flatten again for easier indexing    

    for t in range(T-1):
        max_ix =  int(ix_tracker[T-2-t,max_ix].item())
        unravel_max_ix = np.unravel_index(max_ix, kr_shape)
        opt_augstateix_list =  [np.array(unravel_max_ix).tolist()] + opt_augstateix_list

    opt_state_list = [state_ix[k[0]] for k in opt_augstateix_list]
    if debug:
        return opt_state_list, opt_augstateix_list, val, ix_tracker
    
    return opt_state_list, opt_augstateix_list