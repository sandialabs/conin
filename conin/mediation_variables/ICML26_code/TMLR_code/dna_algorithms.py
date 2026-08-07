# Compliation of all capabilities in dna exemplar
import numpy as np
import torch
from itertools import chain
import copy
from munch import Munch

#=====================================
#Preprocessing
#=====================================
def create_cst_params(cst, hidden_states, dtype = torch.float32, device = 'cpu'):
    m_states = cst.m_states
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


def convertTensor_list(hmm, cst_list, dtype = torch.float16, device = 'cpu', hmm_params = None, return_ix = False):
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
        init_mat, eval_mat, upd_mat = create_cst_params(cst, hmm.states, dtype = dtype, device = device)
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

def compute_emitweights_missing(obs_dict,hmm, dtype = torch.float32, device = 'cpu'):
    '''
    obs_dict is a dictionary t:emission, where it lists only the observed emissions. 
    creates a custom dictionary object with a fallback value for queried keys not in the dictionary.
    serves as a drop-in replacement for current code.
    '''
    class FallbackDict(dict):
        def __init__(self, default, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.default = default
    
        def __missing__(self, key):
            return self.default

    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Compute emissions weights for easier access
    emit_weights = FallbackDict(default = torch.ones(len(hmm.states)).type(dtype).to(device))
    for t in obs_dict.keys():
        val = np.array([hmm.eprob[k,obs_dict[t]] for k in hmm.states])
        val = torch.from_numpy(val).type(dtype).to(device)
        emit_weights[t] = val

    return emit_weights

def hmm2numpy(hmm, ix_list = None, return_ix = False):
    '''
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    '''
    #Initialize and convert all quantities  to np.arrays

    if ix_list:
        state_ix, emit_ix = ix_list
    else:
        state_ix = {s: i for i, s in enumerate(hmm.states)}
        emit_ix = {s: i for i, s in enumerate(hmm.emits)}

    K = len(state_ix)
    M = len(emit_ix)
    #Compute the hmm parameters
    tmat = np.zeros((K,K))
    init_prob = np.zeros(K)

    emat = np.zeros((K,M))

    #Initial distribution. 
    for i in hmm.states:
        if i not in hmm.initprob:
            continue
        init_prob[state_ix[i]] = hmm.initprob[i]

    #Transition matrix
    for i in hmm.states:
        for j in hmm.states:
            if (i,j) not in hmm.tprob:
                continue
            tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]

    
    #Emission matrix
    for i in hmm.states:
        for m in hmm.emits:
            if (i,m) not in hmm.eprob:
                continue
            emat[state_ix[i],emit_ix[m]] = hmm.eprob[i,m]

    hmm_params = [init_prob, tmat, emat]

    if return_ix:
        return hmm_params, [state_ix, emit_ix] 
    return hmm_params

def random_draw(p, rng=None):
    '''
    p is a 1D np array.
    single random draw from probability vector p and encode as 1-hot.
    '''
    if rng is None:
        rng = np.random.default_rng()

    n = len(p)
    p = p / p.sum()
    draw = rng.choice(n, p=p)
    one_hot = np.zeros(n, dtype=int)
    one_hot[draw] = 1
    return one_hot

def single_simulation(hmm, min_time = 0, stay=3, pro_before= 10, ix_list=None, rng = None):
    '''
    Draws from hmm with addition constraint that we stay in each state for at least duration "stay"
    pro_before sets the maximum time horizon that promoter must occur by.
    '''
    if rng is None:
        rng = np.random.default_rng()

    # Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, ix_list=ix_list, return_ix=True) 
    init_prob, tmat, emat = hmm_params

    # Prepare dictionary for converting one_hot back to states
    state_ix, emit_ix = ix_list
    state_ix = {v: k for k, v in state_ix.items()}
    emit_ix = {v: k for k, v in emit_ix.items()}


    # Generate (X1,Y1)
    x_curr = random_draw(init_prob, rng)
    current_state = state_ix[np.argmax(x_curr)] # convert one-hot back to state
    x_list = [current_state] 
    emit_dist = x_curr @ emat
    y_curr = random_draw(emit_dist, rng)
    y_list = [emit_ix[np.argmax(y_curr)]]

    x_prev = x_curr

    #Initialize visit_trackers
    visit_pro = current_state == 'pro'
    visit_dis = current_state == 'dis'
    visit_enh = current_state == 'enh'

    dis_visits = int(current_state == 'dis')
    
    # Initialize state stay counter
    stay_counter = 1


    # Generate rest
    itr = 1 #iteration counter
    while current_state != 'end':
        # By Markov property, just clamp to current stay until stay for required time
        if stay_counter < stay:
            stay_counter += 1
        else:
            # Transition to a new state
            x_curr = random_draw(x_prev @ tmat, rng)
            if np.argmax(x_prev) != np.argmax(x_curr):
                stay_counter = 1  # Reset stay counter for the new state
                current_state = state_ix[np.argmax(x_curr)]
                emit_dist = x_curr @ emat
                x_prev = x_curr


                #Update visit_trackers
                visit_pro = visit_pro or current_state == 'pro'
                visit_dis = visit_dis or current_state == 'dis'
                visit_enh = visit_enh or current_state == 'enh'

                if current_state == 'dis':
                #this condition already assumes transition to new state, so records new dis region.
                    dis_visits += 1

        #Constraints
         #check we hit promoter by pro_before 

        itr += 1
        
        if itr == int(pro_before) and (not visit_pro):
            return False

        #pro < dis < enh

        if (not visit_pro) and visit_dis:
            return False

        if (not visit_dis) and visit_enh:
            return False

        y_curr = random_draw(emit_dist, rng)
        
        x_list.append(current_state)
        y_list.append(emit_ix[np.argmax(y_curr)])

    #check if only one dis region
    if dis_visits != 1:
        return False

    if not visit_enh:
        return False

    
    return x_list, y_list

def simulation(hmm, min_time = 0, stay=5, pro_before=30, ix_list=None, max_attempts=1000, rng = None):
    '''
    Repeatedly calls the simulation function until a valid full run is generated.
    Returns the first valid simulation (list of states and emissions).
    If no valid simulation is found within max_attempts, raises an exception.
    '''
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(max_attempts):
        result = single_simulation(hmm, min_time, stay=stay, pro_before=pro_before, ix_list=ix_list, rng = rng)
        if result is not False:
            return result  # Return the valid simulation

    raise RuntimeError(f"Failed to generate a valid simulation after {max_attempts} attempts.")


#=====================================
# Inference
#=====================================

def Viterbi_torch_list(hmm, cst_list, obs, pro_before = 30, dtype = torch.float32,  device = 'cpu', debug = False, num_corr = 0, hmm_params = None):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, dtype = dtype, \
                                                               device = device, return_ix = True, hmm_params = hmm_params)   
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

    V = torch.einsum(emit_weights[0], [0], init_prob, [0], *init_ind_list, kr_indices) #(K,C1,C2,C3,...)
    V = V/(V.max() + num_corr) #normalize for numerical stability
    val[0] = V.cpu()
    
    for t in range(1,T):
        # return kr_indices, ind_list, dims_list, C
        # V = torch.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = torch.einsum(val[t-1].to(device), js_indices, tmat, [C+1,0], *ind_list, list(range(2*C + 2)))
        V = V.reshape(tuple(kr_shape) + (-1,)) #colapse  the predecessor indices js into a single dim
        V = V/(V.max() + num_corr)
        max_ix = torch.argmax(V, axis = -1, keepdims = True)
        ix_tracker[t-1] = max_ix.squeeze(-1)
        V = torch.take_along_dim(V, max_ix, axis=-1).squeeze(-1)
        # if t == T:
        #     # val[t] = torch.einsum('k,kr,kr -> kr',emit_weights[t],final_ind,V)
        #     val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices,*final_ind_list, kr_indices).cpu()
        # else:
        #     # val[t] = torch.einsum('k,kr -> kr', emit_weights[t],V)
        #     val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices, kr_indices).cpu()
        if t == pro_before - 1:
            # Evaluate only the first constraint at time = 30
            val[t] = torch.einsum(emit_weights[t], [0], V, kr_indices, *final_ind_list[:2], kr_indices).cpu()
        elif t == T -1:
            # Evaluate all constraints at the last time
            val[t] = torch.einsum(emit_weights[t], [0], V, kr_indices, *final_ind_list[2:], kr_indices).cpu()
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
    
#=====================================
# Sampling: Fixed and Variable Length
#=====================================


def index_sampler(arr):
    """
    Given nonnegative tensor/array "arr", samples indices with probability proportional to their weight.
    """
    arr_flat = arr.reshape(-1)

    if (arr_flat < 0).any():
        raise ValueError("All entries must be nonnegative.")
    
    # torch.multinomial expects weights (need not be normalized)
    flat_idx = torch.multinomial(arr_flat, num_samples=1)

    # Convert flat index back to N-D index
    idx_tuple = torch.unravel_index(flat_idx, arr.shape)
    return idx_tuple




def ffbs_torch_list(hmm, cst_list, length_param, pro_before = 30, dtype = torch.float32,  device = 'cpu', debug = False, hmm_params = None):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    if type(length_param) is int:
        emit_weights = np.ones(length_param)
        T = length_param
    elif isinstance(length_param, list):
        emit_weights = compute_emitweights(length_param, hmm)
        emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)
        T = emit_weights.shape[0]

    elif isinstance(length_param, tuple):
        T, emit_dict = length_param
        emit_weights = compute_emitweights_missing(emit_dict,hmm, dtype = dtype, device = device)

    else:
        raise ValueError('length_param must be either an int, list of observations, or tuple containg length and dictionary of observed times')

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, dtype = dtype, \
                                                               device = device, return_ix = True, hmm_params = hmm_params)   
    tmat, init_prob = hmm_params
    dims_list, init_list,eval_list,upd_list = cst_params_list

    # #Assume that last constraint is the one whose sat time is estimated
    # #Parameters for the other fixed constraints
    # fixed_init, fixed_eval, fixed_upd = init_ind_list[:-2], final_ind_list[:-2], ind_list[:-2]

    # #Parameters for estimated sat time constraint.
    # sat_init, sat_true_eval, sat_upd = init_ind_list[-2:], final_ind_list[-2:], ind_list[-2:]
    # sat_false_eval = [1-sat_true_eval[0], sat_true_eval[1]]
    
    #Viterbi
    K = tmat.shape[0]
    C = len(dims_list)
    
    alpha = torch.empty((T,K) + tuple(dims_list), device = 'cpu')
    
    kr_indices = list(range(C+1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]


    #initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    #indices are kurjvs
    alpha[0] = torch.einsum(emit_weights[0],[0], init_prob, [0],*init_list, kr_indices).cpu()

    #Compute forward messages:
    for t in range(1,T):
        #Common terms, summing over js indices.
        V = torch.einsum(alpha[t-1].to(device), js_indices, tmat, [C+1,0], emit_weights[t],[0], \
                                   *upd_list, kr_indices)

        V = V/V.sum() #stepwise renormalization ok, as alpha dictates each steps sampling weights.

        if t == pro_before - 1:
            alpha[t] = torch.einsum(V, kr_indices, *eval_list[:2], kr_indices).cpu()
            
        elif t == T -1 :
            alpha[t] = torch.einsum(V, kr_indices, *eval_list[2:], kr_indices).cpu()

        else:
            alpha[t] = V #torch.einsum(V, kr_indices, kr_indices).cpu()

        if alpha[t].sum().item() <= 0:
            raise ValueError(f'at step {t} the forward messages sum to 0. Previous message sum is {alpha[t-1].sum()}. \n Time horizon is {T}')



    # return alpha
    #Sample paths:
    upd_tensor_list = upd_list[::2] #extract just the upd tensor, not the indices
    ix_list = [index_sampler(alpha[-1])]
    
    for t in range(T-2,-1,-1):
        last_ix = ix_list[-1]
        
        transition_row_list =  [tmat.cpu()[:,last_ix[0].item()], [0]]
        transition_row_list += list(chain.from_iterable((upd.cpu()[last_ix[0].item(),last_ix[ix].item(),:], [ix]) for \
                                       ix, upd in enumerate(upd_tensor_list, start=1)))

        probs = torch.einsum(*transition_row_list, kr_indices)
        probs = probs * alpha[t]
        ix_list.append(index_sampler(probs))

    #Decode.
    state_ix = {v:k for k,v in state_ix.items()} #flip, so indices map to states

    sampled_path = [state_ix[s[0].item()] for s in ix_list]
    sampled_path.reverse()

    return sampled_path

def variable_length_sampling(hmm, time_cst, sample_cst, length_param, min_time = 0, pro_before = 30, dtype = torch.float32,  device = 'cpu', debug = False, hmm_params = None):
    '''
    Samples the stopping time, then samples a feasible path.

    time_cst is the constraint list used to sample stopping time: last constraint should be stopping time
    sample_cst is constraint list for sampling path.
    '''
    probs, _, _ = satTime_torch_list(hmm, time_cst, length_param, min_time = min_time, pro_before = pro_before, dtype = dtype,  device = device)
    probs = probs/probs.sum()
    length = torch.multinomial(probs, num_samples=1).item() + min_time + 1 #add 1 to account for 0 indexing
    sample_path = ffbs_torch_list(hmm, sample_cst, length_param = (length,length_param[1]), pro_before = pro_before, dtype = dtype,  device = device)
    
    return sample_path, length
    
#=====================================
# Satisfaction Time
#=====================================
def satTime_torch_list(hmm, cst_list, length_param, min_time = 0, pro_before = 30, dtype = torch.float32,  device = 'cpu', debug = False, hmm_params = None):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    if type(length_param) is int:
        emit_weights = np.ones(length_param)
        T = length_param
    elif isinstance(length_param, list):
        emit_weights = compute_emitweights(length_param, hmm)
        emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)
        T = emit_weights.shape[0]

    elif isinstance(length_param, tuple):
        T, emit_dict = length_param
        emit_weights = compute_emitweights_missing(emit_dict,hmm, dtype = dtype, device = device)

    else:
        raise ValueError('length_param must be either an int, list of observations, or tuple containg length and dictionary of observed times')

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, dtype = dtype, \
                                                               device = device, return_ix = True, hmm_params = hmm_params)   
    tmat, init_prob = hmm_params
    dims_list, init_ind_list,final_ind_list,ind_list = cst_params_list

    #Assume that last constraint is the one whose sat time is estimated
    #Parameters for the other fixed constraints
    fixed_init, fixed_eval, fixed_upd = init_ind_list[:-2], final_ind_list[:-2], ind_list[:-2]

    #Parameters for estimated sat time constraint.
    sat_init, sat_true_eval, sat_upd = init_ind_list[-2:], final_ind_list[-2:], ind_list[-2:]
    sat_false_eval = [1-sat_true_eval[0], sat_true_eval[1]]
    
    #Viterbi
    K = tmat.shape[0]
    C = len(dims_list)
    
    alpha = torch.empty((T,K) + tuple(dims_list), device = 'cpu')
    gamma = torch.empty(alpha.shape, device = 'cpu')
    #beta doesn't need to track mediation space of estimate sat time. Dummy dim for last.
    beta = torch.empty((T, K) + tuple(dims_list[:-1]), device='cpu')
    sat_probs = torch.empty((T,))
    
    kr_indices = list(range(C+1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]


    #initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    #indices are kurjvs
    gamma[0] = torch.einsum(emit_weights[0],[0], init_prob, [0],*init_ind_list, *sat_false_eval, kr_indices).cpu()
    alpha[0] = torch.einsum(emit_weights[0],[0], init_prob, [0],*init_ind_list, *sat_true_eval, kr_indices).cpu()
    beta[-1] = 1
    beta[-2] = torch.einsum(tmat,[C+1, 0], *fixed_upd, *fixed_eval[2:], emit_weights[-1], [0], js_indices[:-1]).cpu()

    fwd_norm = torch.zeros((T,))
    bck_norm = torch.zeros((T,))

    fwd_norm[0] = 0
    bck_norm[-1] = 0
    bck_norm[-2] = 0

    #the total normalization at alpha[t] is product of all up-to-present normalizations
    log_running_norm = 0
    #Compute forward messages:
    for t in range(1,T):
        #Common terms, summing over js indices.
        V = torch.einsum(gamma[t-1].to(device), js_indices, tmat, [C+1,0], emit_weights[t],[0], \
                                   *ind_list, kr_indices)

        norm = V.sum()
        V = V/norm

        if norm.item() <= 0.:
            raise ValueError(f'sums to 0! at time {t} on forward pass')
            
        log_running_norm += torch.log(norm)
        fwd_norm[t] = log_running_norm
        if torch.abs(log_running_norm) > 300:
            fwd_norm[:t+1] = fwd_norm[:t+1] - log_running_norm #renormalize for stability
            log_running_norm = 0
        
        if t == pro_before:
            gamma[t] = torch.einsum(V, kr_indices, *sat_false_eval, *fixed_upd[:2], kr_indices).cpu()
            alpha[t] = torch.einsum(V, kr_indices, *sat_true_eval, *fixed_upd[:2], kr_indices).cpu()
            
        elif t == T -1 :
            gamma[t] = torch.einsum(V, kr_indices, *sat_false_eval, *fixed_eval[2:], kr_indices).cpu()
            alpha[t] = torch.einsum(V, kr_indices, *sat_true_eval, *fixed_eval[2:], kr_indices).cpu()
            
        else:
            gamma[t] = torch.einsum(V, kr_indices, *sat_false_eval, kr_indices).cpu()
            alpha[t] = torch.einsum(V, kr_indices, *sat_true_eval, kr_indices).cpu()
            



    #Compute backward messages
    log_running_norm = 0
    for t in range(T-3,-1,-1):
        if t == (pro_before - 1):
            beta[t] = torch.einsum(beta[t+1].to(device), kr_indices[:-1], tmat, [C+1, 0], *fixed_upd, \
                               emit_weights[t+1], [0], *fixed_eval[:2], js_indices[:-1]).cpu()
        else:
            beta[t] = torch.einsum(beta[t+1].to(device), kr_indices[:-1], tmat, [C+1, 0], *fixed_upd, \
                               emit_weights[t+1], [0], js_indices[:-1]).cpu()
        norm = beta[t].sum()
        if norm.item() <= 0.:
            raise ValueError(f'sums to 0! at time {t} on backward pass')
        beta[t] = beta[t]/norm
        # running_norm *= norm
        # bck_norm[t] = running_norm
        # if running_norm <= 1e-7:
        #     bck_norm[t:] = bck_norm[t:] / running_norm #renormalize for stability
        #     running_norm = 1
            
        log_running_norm += torch.log(norm)
        bck_norm[t] = log_running_norm
        if torch.abs(log_running_norm) > 300:
            bck_norm[t:] = bck_norm[t:] - log_running_norm #renormalize for stability
            log_running_norm = 0

    #Compute moments
    for t in range(T):
        alpha_msg = alpha[t].to(device)
        beta_msg = beta[t].to(device)
        # sat_probs[t] = torch.einsum('...i,... ->', alpha_msg, beta_msg)
        sat_probs[t] = torch.einsum(alpha_msg, kr_indices, beta_msg, kr_indices[:-1], [])

    #Center the log normalization constants for stabilization
    fwd_norm = fwd_norm - fwd_norm.mean()
    bck_norm = bck_norm - bck_norm.mean()
    
    # return sat_probs, fwd_norm, bck_norm

    # sat_probs = sat_probs/sat_probs.sum() #normalize once for numerical stability
    sat_probs = sat_probs * torch.exp(fwd_norm + bck_norm)
    sat_probs = sat_probs[min_time:]
    sat_probs = sat_probs/sat_probs.sum()

    return sat_probs, alpha, beta

#=====================================
# Satisfaction Probability
#=====================================

def satprob_torch_list(hmm, cst_list, obs, pro_before = 30, dtype = torch.float32,  device = 'cpu', debug = False, num_corr = 0, hmm_params = None):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, dtype = dtype, \
                                                               device = device, return_ix = True, hmm_params = hmm_params)   
    tmat, init_prob = hmm_params
    dims_list, init_list,eval_list,upd_list = cst_params_list

    #Assume that last constraint is the one whose sat time is estimated
    #Parameters for the other fixed constraints
    fixed_init, fixed_eval, fixed_upd = init_list[:-2], eval_list[:-2], upd_list[:-2]

    #Parameters for estimated sat time constraint.
    sat_init, sat_true_eval, sat_upd = init_list[-2:], eval_list[-2:], upd_list[-2:]
    sat_false_eval = [1-sat_true_eval[0], sat_true_eval[1]]
    
    #Viterbi
    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)
    
    
    kr_indices = list(range(C+1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]


    #initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    #indices are kurjvs
    alpha = torch.einsum(emit_weights[0],[0], init_prob, [0],*fixed_init, *sat_init, kr_indices)
    alpha = alpha/(alpha.sum()) #normalize for stability
    #Compute forward messages:
    for t in range(1,T-1):
        #Common terms, summing over js indices.
        if t == pro_before - 1:
            # alpha[t] = torch.einsum(V, kr_indices, *fixed_upd[:2], kr_indices).cpu()
            
            alpha = torch.einsum(alpha, js_indices, tmat, [C+1,0], emit_weights[t],[0], \
                                       *upd_list, *fixed_eval[:2], kr_indices)
        else:
            alpha = torch.einsum(alpha, js_indices, tmat, [C+1,0], emit_weights[t],[0], \
                                       *upd_list, kr_indices)

        alpha = alpha/(alpha.max() + num_corr) #normalize for stability
    #compute final probs
    alpha_true = torch.einsum(alpha.to(device), js_indices, tmat, [C+1,0], emit_weights[T-1],[0], \
                               *upd_list, *fixed_eval[2:], *sat_true_eval, kr_indices)
    alpha_false = torch.einsum(alpha.to(device), js_indices, tmat, [C+1,0], emit_weights[T-1],[0], \
                               *upd_list, *fixed_eval[2:], *sat_false_eval, kr_indices)

    probs = torch.tensor([alpha_true.sum().cpu().item(), alpha_false.sum().cpu().item()] )
    probs = probs/probs.sum()
    
    return probs #, alpha_true, alpha_false

# =====================================
# Learning: Helper Functions
# =====================================

def _build_emission_support_mask(hmm, end_state='end', end_emission='N'):
    """
    Build a binary mask of allowed emission entries.

    Rules:
      - state == end_state: only end_emission allowed
      - all other states: end_emission forbidden, all other emissions allowed
    """
    states = list(hmm.states)
    emits = list(hmm.emits)

    state_ix = {s: i for i, s in enumerate(states)}
    emit_ix = {m: i for i, m in enumerate(emits)}

    K = len(states)
    M = len(emits)

    if end_state not in state_ix:
        raise ValueError(f"end_state='{end_state}' not found in hmm.states")
    if end_emission not in emit_ix:
        raise ValueError(f"end_emission='{end_emission}' not found in hmm.emits")

    mask = np.ones((K, M), dtype=np.float64)

    end_k = state_ix[end_state]
    end_m = emit_ix[end_emission]

    mask[end_k, :] = 0.0
    mask[end_k, end_m] = 1.0

    for k in range(K):
        if k != end_k:
            mask[k, end_m] = 0.0

    return mask, state_ix, emit_ix


def _normalize_probvec(v, eps=0.0):
    v = np.asarray(v, dtype=np.float64)
    if eps > 0:
        v = v + eps
    s = v.sum()
    if s <= 0:
        raise ValueError("Cannot normalize vector with nonpositive sum.")
    return v / s


def _normalize_rows(mat, eps=0.0):
    mat = np.asarray(mat, dtype=np.float64).copy()
    if eps > 0:
        mat = mat + eps
    row_sums = mat.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0
    if np.any(bad):
        mat[bad, :] = 1.0
        row_sums = mat.sum(axis=1, keepdims=True)
    return mat / row_sums


def _normalize_rows_on_support(mat, support_mask, eps=0.0):
    """
    Normalize rows only over supported entries.
    Forbidden entries are set to 0 exactly.
    """
    mat = np.asarray(mat, dtype=np.float64).copy()
    support_mask = np.asarray(support_mask, dtype=np.float64)

    mat = mat * support_mask
    if eps > 0:
        mat = mat + eps * support_mask

    row_sums = mat.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0

    if np.any(bad):
        for i in np.where(bad)[0]:
            support = support_mask[i] > 0
            if not np.any(support):
                raise ValueError(f"Row {i} has empty emission support.")
            mat[i, :] = 0.0
            mat[i, support] = 1.0 / support.sum()
        row_sums = mat.sum(axis=1, keepdims=True)

    out = mat / row_sums
    out = out * support_mask
    return out


def _update_emission_matrix_with_end_constraint(
    emit_counts_total,
    hmm,
    pseudocount=1e-8,
    end_state='end',
    end_emission='N',
):
    support_mask, _, _ = _build_emission_support_mask(
        hmm,
        end_state=end_state,
        end_emission=end_emission,
    )
    return _normalize_rows_on_support(
        emit_counts_total,
        support_mask,
        eps=pseudocount,
    )


def randomize_hmm_like(hmm, rng=None, alpha=1.0):
    """
    Create a new HMM object with the same states/emissions as `hmm`,
    but with randomly initialized initprob, tprob, and eprob.

    Emission initialization respects:
      - 'end' emits only 'N'
      - other states emit only over non-'N'
    """
    if rng is None:
        rng = np.random.default_rng()

    states = list(hmm.states)
    emits = list(hmm.emits)

    K = len(states)
    M = len(emits)

    new_hmm = Munch()
    new_hmm.states = states
    new_hmm.emits = emits

    init_vec = rng.dirichlet(alpha * np.ones(K))
    new_hmm.initprob = {
        states[i]: float(init_vec[i]) for i in range(K)
    }

    tmat = np.array([rng.dirichlet(alpha * np.ones(K)) for _ in range(K)])
    new_hmm.tprob = {
        (states[i], states[j]): float(tmat[i, j])
        for i in range(K) for j in range(K)
    }

    support_mask, _, _ = _build_emission_support_mask(new_hmm)

    emat = np.zeros((K, M), dtype=np.float64)
    for i in range(K):
        support = np.where(support_mask[i] > 0)[0]
        draw = rng.dirichlet(alpha * np.ones(len(support)))
        emat[i, support] = draw

    new_hmm.eprob = {
        (states[i], emits[m]): float(emat[i, m])
        for i in range(K) for m in range(M)
    }

    return new_hmm


# =====================================
# Unconstrained Learning
# =====================================

def _hmm_to_torch_params(hmm, ix_list=None, dtype=torch.float64, device='cpu'):
    """
    Torch version of hmm2numpy outputs.
    Returns:
      init_prob: (K,)
      tmat: (K,K)
      emat: (K,M)
      ix_list: [state_ix, emit_ix]
    """
    hmm_params, ix_list = hmm2numpy(hmm, ix_list=ix_list, return_ix=True)
    init_prob_np, tmat_np, emat_np = hmm_params

    init_prob = torch.tensor(init_prob_np, dtype=dtype, device=device)
    tmat = torch.tensor(tmat_np, dtype=dtype, device=device)
    emat = torch.tensor(emat_np, dtype=dtype, device=device)

    return init_prob, tmat, emat, ix_list


def unconstrained_e_step_counts(hmm, obs, ix_list=None, dtype=torch.float64, device='cpu'):
    """
    Count-only forward-backward for a standard HMM.

    Returns
    -------
    init_counts : torch.Tensor, shape (K,)
    trans_counts : torch.Tensor, shape (K,K)
    emit_counts : torch.Tensor, shape (K,M)
    loglik : float
    ix_list : list
    """
    init_prob, tmat, emat, ix_list = _hmm_to_torch_params(
        hmm, ix_list=ix_list, dtype=dtype, device=device
    )
    state_ix, emit_ix = ix_list

    obs_idx = torch.tensor([emit_ix[o] for o in obs], dtype=torch.long, device=device)

    T = len(obs)
    K = init_prob.shape[0]
    M = emat.shape[1]

    alpha = torch.empty((T, K), dtype=dtype, device=device)
    beta = torch.empty((T, K), dtype=dtype, device=device)
    scales = torch.empty(T, dtype=dtype, device=device)

    # Forward
    alpha[0] = init_prob * emat[:, obs_idx[0]]
    s0 = alpha[0].sum()
    if s0.item() <= 0:
        raise ValueError("Forward message at t=0 sums to 0.")
    alpha[0] /= s0
    scales[0] = s0

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ tmat) * emat[:, obs_idx[t]]
        st = alpha[t].sum()
        if st.item() <= 0:
            raise ValueError(f"Forward message at t={t} sums to 0.")
        alpha[t] /= st
        scales[t] = st

    loglik = torch.log(scales).sum().item()

    # Backward
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = tmat @ (emat[:, obs_idx[t + 1]] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(dim=1, keepdim=True)

    init_counts = gamma[0]
    trans_counts = torch.zeros((K, K), dtype=dtype, device=device)
    emit_counts = torch.zeros((K, M), dtype=dtype, device=device)

    for t in range(T - 1):
        numer = (
            alpha[t][:, None]
            * tmat
            * (emat[:, obs_idx[t + 1]] * beta[t + 1])[None, :]
        )
        denom = numer.sum()
        if denom.item() <= 0:
            raise ValueError(f"Transition posterior at t={t} sums to 0.")
        trans_counts += numer / denom

    for t in range(T):
        emit_counts[:, obs_idx[t]] += gamma[t]

    return init_counts, trans_counts, emit_counts, loglik, ix_list


def unconstrained_forward_backward(hmm, obs, ix_list=None, dtype=torch.float64, device='cpu'):
    """
    Compatibility wrapper returning gamma / xi like the older version.
    Less efficient than unconstrained_e_step_counts, but kept for API continuity.
    """
    init_prob, tmat, emat, ix_list = _hmm_to_torch_params(
        hmm, ix_list=ix_list, dtype=dtype, device=device
    )
    state_ix, emit_ix = ix_list

    obs_idx = torch.tensor([emit_ix[o] for o in obs], dtype=torch.long, device=device)

    T = len(obs)
    K = init_prob.shape[0]

    alpha = torch.empty((T, K), dtype=dtype, device=device)
    beta = torch.empty((T, K), dtype=dtype, device=device)
    scales = torch.empty(T, dtype=dtype, device=device)

    alpha[0] = init_prob * emat[:, obs_idx[0]]
    s0 = alpha[0].sum()
    if s0.item() <= 0:
        raise ValueError("Forward message at t=0 sums to 0.")
    alpha[0] /= s0
    scales[0] = s0

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ tmat) * emat[:, obs_idx[t]]
        st = alpha[t].sum()
        if st.item() <= 0:
            raise ValueError(f"Forward message at t={t} sums to 0.")
        alpha[t] /= st
        scales[t] = st

    loglik = torch.log(scales).sum().item()

    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = tmat @ (emat[:, obs_idx[t + 1]] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(dim=1, keepdim=True)

    xi = torch.empty((T - 1, K, K), dtype=dtype, device=device)
    for t in range(T - 1):
        numer = (
            alpha[t][:, None]
            * tmat
            * (emat[:, obs_idx[t + 1]] * beta[t + 1])[None, :]
        )
        denom = numer.sum()
        if denom.item() <= 0:
            raise ValueError(f"Transition posterior at t={t} sums to 0.")
        xi[t] = numer / denom

    return gamma.cpu().numpy(), xi.cpu().numpy(), loglik, ix_list, [
        init_prob.detach().cpu().numpy(),
        tmat.detach().cpu().numpy(),
        emat.detach().cpu().numpy(),
    ]


def baum_welch_unconstrained(
    hmm,
    obs_batch,
    max_iter=50,
    tol=1e-6,
    pseudocount=1e-8,
    verbose=False,
    dtype=torch.float64,
    device='cpu',
):
    """
    Optimized Baum-Welch / EM for a batch of sequences.
    Uses count-only E-step.
    """
    hmm = copy.deepcopy(hmm)

    state_ix = {s: i for i, s in enumerate(hmm.states)}
    emit_ix = {m: i for i, m in enumerate(hmm.emits)}
    inv_state_ix = {i: s for s, i in state_ix.items()}
    inv_emit_ix = {i: m for m, i in emit_ix.items()}

    K = len(hmm.states)
    M = len(hmm.emits)

    history = []
    ix_list = [state_ix, emit_ix]

    for it in range(max_iter):
        init_counts_total = torch.zeros(K, dtype=dtype, device=device)
        trans_counts_total = torch.zeros((K, K), dtype=dtype, device=device)
        emit_counts_total = torch.zeros((K, M), dtype=dtype, device=device)
        total_loglik = 0.0

        for obs in obs_batch:
            init_counts, trans_counts, emit_counts, loglik, _ = unconstrained_e_step_counts(
                hmm,
                obs,
                ix_list=ix_list,
                dtype=dtype,
                device=device,
            )
            init_counts_total += init_counts
            trans_counts_total += trans_counts
            emit_counts_total += emit_counts
            total_loglik += loglik

        history.append(total_loglik)
        if verbose:
            print(f"EM iter {it:3d}  loglik = {total_loglik:.10f}")

        if it > 0 and abs(history[-1] - history[-2]) < tol:
            break

        init_prob_new = _normalize_probvec(
            init_counts_total.detach().cpu().numpy(),
            eps=pseudocount
        )
        tmat_new = _normalize_rows(
            trans_counts_total.detach().cpu().numpy(),
            eps=pseudocount
        )
        emat_new = _update_emission_matrix_with_end_constraint(
            emit_counts_total.detach().cpu().numpy(),
            hmm,
            pseudocount=pseudocount,
            end_state='end',
            end_emission='N',
        )

        hmm.initprob = {
            inv_state_ix[i]: float(init_prob_new[i])
            for i in range(K)
        }
        hmm.tprob = {
            (inv_state_ix[i], inv_state_ix[j]): float(tmat_new[i, j])
            for i in range(K) for j in range(K)
        }
        hmm.eprob = {
            (inv_state_ix[i], inv_emit_ix[m]): float(emat_new[i, m])
            for i in range(K) for m in range(M)
        }

    return hmm, history


# =====================================
# Constrained Learning
# =====================================

def _prepare_constrained_static_params(hmm, cst_list, dtype=torch.float64, device='cpu'):
    """
    Build static constraint tensors once.
    """
    hmm_params, cst_params_list, state_ix = convertTensor_list(
        hmm,
        cst_list,
        dtype=dtype,
        device=device,
        return_ix=True,
        hmm_params=None,
    )
    _, _ = hmm_params
    dims_list, init_list, eval_list, upd_list = cst_params_list
    return cst_params_list, state_ix


def _current_hmm_torch_params_only(hmm, state_ix, dtype=torch.float64, device='cpu'):
    """
    Rebuild only the learned HMM params in torch form:
      tmat: (K,K)
      init_prob: (K,)
    """
    K = len(hmm.states)
    tmat = torch.zeros((K, K), dtype=dtype, device=device)
    init_prob = torch.zeros(K, dtype=dtype, device=device)

    for s in hmm.states:
        i = state_ix[s]
        init_prob[i] = hmm.initprob[s]
        for t in hmm.states:
            j = state_ix[t]
            tmat[i, j] = hmm.tprob[s, t]

    return tmat, init_prob


def constrained_e_step_counts(
    hmm,
    cst_list,
    obs,
    pro_before=30,
    dtype=torch.float64,
    device='cpu',
    static_cst=None,
):
    """
    Count-only forward-backward for the constrained augmented HMM.

    Returns
    -------
    init_counts : torch.Tensor, shape (K,)
    trans_counts : torch.Tensor, shape (K,K)
    emit_counts : torch.Tensor, shape (K,M)
    loglik : float
    """
    if static_cst is None:
        cst_params_list, state_ix = _prepare_constrained_static_params(
            hmm, cst_list, dtype=dtype, device=device
        )
    else:
        cst_params_list, state_ix = static_cst

    tmat, init_prob = _current_hmm_torch_params_only(
        hmm, state_ix, dtype=dtype, device=device
    )

    dims_list, init_list, eval_list, upd_list = cst_params_list

    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).to(device=device, dtype=dtype)

    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)
    M = len(hmm.emits)

    emit_ix = {m: i for i, m in enumerate(hmm.emits)}
    obs_idx = torch.tensor([emit_ix[o] for o in obs], dtype=torch.long, device=device)

    kr_indices = list(range(C + 1))
    js_indices = [i + C + 1 for i in kr_indices]
    aug_shape = (K,) + tuple(dims_list)

    alpha = torch.empty((T,) + aug_shape, dtype=dtype, device=device)
    beta = torch.empty((T,) + aug_shape, dtype=dtype, device=device)
    scales = torch.empty(T, dtype=dtype, device=device)

    def apply_eval_factors_current(tensor, t):
        if t == pro_before - 1 and len(eval_list) >= 2:
            tensor = torch.einsum(tensor, kr_indices, *eval_list[:2], kr_indices)
        if t == T - 1 and len(eval_list) > 2:
            tensor = torch.einsum(tensor, kr_indices, *eval_list[2:], kr_indices)
        return tensor

    # Forward
    a0 = torch.einsum(
        emit_weights[0], [0],
        init_prob, [0],
        *init_list,
        kr_indices
    )
    a0 = apply_eval_factors_current(a0, 0)

    s0 = a0.sum()
    if s0.item() <= 0:
        raise ValueError("Forward message at t=0 sums to 0.")
    alpha[0] = a0 / s0
    scales[0] = s0

    for t in range(1, T):
        V = torch.einsum(
            alpha[t - 1], js_indices,
            tmat, [C + 1, 0],
            emit_weights[t], [0],
            *upd_list,
            kr_indices
        )
        V = apply_eval_factors_current(V, t)

        st = V.sum()
        if st.item() <= 0:
            raise ValueError(f"Forward message at t={t} sums to 0.")
        alpha[t] = V / st
        scales[t] = st

    loglik = torch.log(scales).sum().item()

    # Backward
    beta[-1] = torch.ones(aug_shape, dtype=dtype, device=device)

    for t in range(T - 2, -1, -1):
        next_msg = torch.einsum(
            beta[t + 1], kr_indices,
            emit_weights[t + 1], [0],
            kr_indices
        )
        next_msg = apply_eval_factors_current(next_msg, t + 1)

        B = torch.einsum(
            next_msg, kr_indices,
            tmat, [C + 1, 0],
            *upd_list,
            js_indices
        )
        beta[t] = B / scales[t + 1]

    # gamma over augmented states
    gamma = alpha * beta
    gamma /= gamma.reshape(T, -1).sum(dim=1).reshape((T,) + (1,) * (C + 1))

    # Marginal gamma over hidden state only: (T,K)
    if gamma.ndim > 2:
        gamma_x = gamma.sum(dim=tuple(range(2, gamma.ndim)))
    else:
        gamma_x = gamma

    init_counts = gamma_x[0]
    trans_counts = torch.zeros((K, K), dtype=dtype, device=device)
    emit_counts = torch.zeros((K, M), dtype=dtype, device=device)

    # Accumulate emissions directly from gamma_x
    for t in range(T):
        emit_counts[:, obs_idx[t]] += gamma_x[t]

    # Accumulate hidden-state transition counts directly without storing full xi
    for t in range(T - 1):
        right_msg = torch.einsum(
            beta[t + 1], kr_indices,
            emit_weights[t + 1], [0],
            kr_indices
        )
        right_msg = apply_eval_factors_current(right_msg, t + 1)

        Xi_t = torch.einsum(
            alpha[t], js_indices,
            tmat, [C + 1, 0],
            right_msg, kr_indices,
            *upd_list,
            js_indices + kr_indices
        )

        denom = Xi_t.sum()
        if denom.item() <= 0:
            raise ValueError(f"Xi tensor at t={t} sums to 0.")
        Xi_t = Xi_t / denom

        # Sum out all auxiliary dimensions, leaving (K,K)
        n_aux = len(dims_list)
        if n_aux > 0:
            left_aux_axes = tuple(range(1, 1 + n_aux))
            right_aux_axes = tuple(range(2 + n_aux, 2 + 2 * n_aux))
            xi_x_t = Xi_t.sum(dim=left_aux_axes + right_aux_axes)
        else:
            xi_x_t = Xi_t

        trans_counts += xi_x_t

    return init_counts, trans_counts, emit_counts, loglik


def constrained_forward_backward_augmented(
    hmm,
    cst_list,
    obs,
    pro_before=30,
    dtype=torch.float64,
    device='cpu',
    hmm_params=None,
):
    """
    Compatibility version returning gamma / xi.
    This keeps tensors on device, but is not as efficient as constrained_e_step_counts.
    """
    # static tensors
    hmm_params_full, cst_params_list, state_ix = convertTensor_list(
        hmm,
        cst_list,
        dtype=dtype,
        device=device,
        return_ix=True,
        hmm_params=hmm_params,
    )
    tmat, init_prob = hmm_params_full
    dims_list, init_list, eval_list, upd_list = cst_params_list

    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)

    kr_indices = list(range(C + 1))
    js_indices = [i + C + 1 for i in kr_indices]
    aug_shape = (K,) + tuple(dims_list)

    alpha = torch.empty((T,) + aug_shape, dtype=dtype, device=device)
    beta = torch.empty((T,) + aug_shape, dtype=dtype, device=device)
    scales = torch.empty(T, dtype=dtype, device=device)

    def apply_eval_factors_current(tensor, t):
        if t == pro_before - 1 and len(eval_list) >= 2:
            tensor = torch.einsum(tensor, kr_indices, *eval_list[:2], kr_indices)
        if t == T - 1 and len(eval_list) > 2:
            tensor = torch.einsum(tensor, kr_indices, *eval_list[2:], kr_indices)
        return tensor

    a0 = torch.einsum(
        emit_weights[0], [0],
        init_prob, [0],
        *init_list,
        kr_indices
    )
    a0 = apply_eval_factors_current(a0, 0)
    s0 = a0.sum()
    if s0.item() <= 0:
        raise ValueError("Forward message at t=0 sums to 0.")
    alpha[0] = a0 / s0
    scales[0] = s0

    for t in range(1, T):
        V = torch.einsum(
            alpha[t - 1], js_indices,
            tmat, [C + 1, 0],
            emit_weights[t], [0],
            *upd_list,
            kr_indices
        )
        V = apply_eval_factors_current(V, t)
        st = V.sum()
        if st.item() <= 0:
            raise ValueError(f"Forward message at t={t} sums to 0.")
        alpha[t] = V / st
        scales[t] = st

    loglik = torch.log(scales).sum().item()

    beta[-1] = torch.ones(aug_shape, dtype=dtype, device=device)

    for t in range(T - 2, -1, -1):
        next_msg = torch.einsum(
            beta[t + 1], kr_indices,
            emit_weights[t + 1], [0],
            kr_indices
        )
        next_msg = apply_eval_factors_current(next_msg, t + 1)

        B = torch.einsum(
            next_msg, kr_indices,
            tmat, [C + 1, 0],
            *upd_list,
            js_indices
        )
        beta[t] = B / scales[t + 1]

    gamma = alpha * beta
    gamma /= gamma.reshape(T, -1).sum(dim=1).reshape((T,) + (1,) * (C + 1))

    xi = torch.empty((T - 1,) + aug_shape + aug_shape, dtype=dtype, device=device)
    all_indices = js_indices + kr_indices

    for t in range(T - 1):
        right_msg = torch.einsum(
            beta[t + 1], kr_indices,
            emit_weights[t + 1], [0],
            kr_indices
        )
        right_msg = apply_eval_factors_current(right_msg, t + 1)

        Xi_t = torch.einsum(
            alpha[t], js_indices,
            tmat, [C + 1, 0],
            right_msg, kr_indices,
            *upd_list,
            all_indices
        )
        denom = Xi_t.sum()
        if denom.item() <= 0:
            raise ValueError(f"Xi tensor at t={t} sums to 0.")
        xi[t] = Xi_t / denom

    return gamma.detach().cpu(), xi.detach().cpu(), loglik, state_ix, dims_list


def baum_welch_constrained(
    hmm,
    cst_list,
    obs_batch,
    pro_before=30,
    max_iter=50,
    tol=1e-6,
    pseudocount=1e-8,
    dtype=torch.float64,
    device='cpu',
    verbose=False,
):
    """
    Optimized constrained Baum-Welch / EM.
    Uses count-only constrained E-step and caches static constraint tensors.
    """
    hmm = copy.deepcopy(hmm)

    state_ix = {s: i for i, s in enumerate(hmm.states)}
    emit_ix = {m: i for i, m in enumerate(hmm.emits)}
    inv_state_ix = {i: s for s, i in state_ix.items()}
    inv_emit_ix = {i: m for m, i in emit_ix.items()}

    K = len(hmm.states)
    M = len(hmm.emits)

    history = []

    # Constraint tensors are static across EM iterations
    static_cst = _prepare_constrained_static_params(
        hmm,
        cst_list,
        dtype=dtype,
        device=device,
    )

    for it in range(max_iter):
        init_counts_total = torch.zeros(K, dtype=dtype, device=device)
        trans_counts_total = torch.zeros((K, K), dtype=dtype, device=device)
        emit_counts_total = torch.zeros((K, M), dtype=dtype, device=device)
        total_loglik = 0.0

        for obs in obs_batch:
            init_counts, trans_counts, emit_counts, loglik = constrained_e_step_counts(
                hmm,
                cst_list,
                obs,
                pro_before=pro_before,
                dtype=dtype,
                device=device,
                static_cst=static_cst,
            )
            init_counts_total += init_counts
            trans_counts_total += trans_counts
            emit_counts_total += emit_counts
            total_loglik += loglik

        history.append(total_loglik)
        if verbose:
            print(f"EM iter {it:3d}  loglik = {total_loglik:.10f}")

        if it > 0 and abs(history[-1] - history[-2]) < tol:
            break

        init_prob_new = _normalize_probvec(
            init_counts_total.detach().cpu().numpy(),
            eps=pseudocount
        )
        tmat_new = _normalize_rows(
            trans_counts_total.detach().cpu().numpy(),
            eps=pseudocount
        )
        emat_new = _update_emission_matrix_with_end_constraint(
            emit_counts_total.detach().cpu().numpy(),
            hmm,
            pseudocount=pseudocount,
            end_state='end',
            end_emission='N',
        )

        hmm.initprob = {
            inv_state_ix[i]: float(init_prob_new[i])
            for i in range(K)
        }
        hmm.tprob = {
            (inv_state_ix[i], inv_state_ix[j]): float(tmat_new[i, j])
            for i in range(K) for j in range(K)
        }
        hmm.eprob = {
            (inv_state_ix[i], inv_emit_ix[m]): float(emat_new[i, m])
            for i in range(K) for m in range(M)
        }

    return hmm, history


    