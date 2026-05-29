# Compliation of all capabilities in dna exemplar
import numpy as np
import torch
from itertools import chain
import copy


# =====================================
# Preprocessing
# =====================================
def create_cst_params(cst, hidden_states, dtype=torch.float32, device="cpu"):
    m_states = cst.m_states
    init = cst.init_fun
    upd = cst.update_fun
    eval_fun = cst.eval_fun

    # returns a (k,s,r) array. k is current hideen. r,s are present/past mediation.
    upd_mat = torch.tensor(
        [[[upd(k, r, s) for s in m_states] for r in m_states] for k in hidden_states],
        dtype=dtype,
        device=device,
    )

    # returns a (k,r) array. k,r are current hidden/mediation states
    init_mat = torch.tensor(
        [[init(k, r) for r in m_states] for k in hidden_states],
        dtype=dtype,
        device=device,
    )

    # return (k,r) array for terminal emission.
    eval_mat = torch.tensor(
        [[eval_fun(k, r) for r in m_states] for k in hidden_states],
        dtype=dtype,
        device=device,
    )

    return init_mat, eval_mat, upd_mat


def convertTensor_list(
    hmm, cst_list, dtype=torch.float16, device="cpu", hmm_params=None, return_ix=False
):
    """
    cst_list is a list of the individual csts.
    """
    # Initialize and convert all quantities  to np.arrays
    hmm = copy.deepcopy(hmm)
    K = len(hmm.states)

    state_ix = {s: i for i, s in enumerate(hmm.states)}

    # Compute the hmm parameters if not provided
    if hmm_params is None:
        tmat = torch.zeros((K, K), dtype=dtype).to(device)
        init_prob = torch.zeros(K, dtype=dtype).to(device)

        for i in hmm.states:
            init_prob[state_ix[i]] = hmm.initprob[i]
            for j in hmm.states:
                tmat[state_ix[i], state_ix[j]] = hmm.tprob[i, j]

        hmm_params = [tmat, init_prob]

    # Compute the cst parameters
    init_list = []
    eval_list = []
    upd_list = []
    dims_list = []
    cst_ix = 0
    C = len(cst_list)

    # indices are (hidden, c_1,....,c_C, hidden, c_1,....,c_C) are augmented messages
    for cst in cst_list:
        cst = copy.deepcopy(cst)
        init_mat, eval_mat, upd_mat = create_cst_params(
            cst, hmm.states, dtype=dtype, device=device
        )
        init_list += [init_mat, [0, cst_ix + 1]]
        eval_list += [eval_mat, [0, cst_ix + 1]]
        upd_list += [upd_mat, [0, cst_ix + 1, cst_ix + C + 2]]
        dims_list.append(len(cst.m_states))
        cst_ix += 1

    cst_params = [dims_list, init_list, eval_list, upd_list]

    if return_ix:
        return hmm_params, cst_params, state_ix
    return hmm_params, cst_params


def compute_emitweights(obs, hmm):
    """
    Separately handles the computation of the
    """
    hmm = copy.deepcopy(hmm)  # protect again in place modification
    T = len(obs)
    K = len(hmm.states)
    # Compute emissions weights for easier access
    emit_weights = np.zeros((T, K))
    for t in range(T):
        emit_weights[t] = np.array([hmm.eprob[k, obs[t]] for k in hmm.states])
    return emit_weights


def compute_emitweights_missing(obs_dict, hmm, dtype=torch.float32, device="cpu"):
    """
    obs_dict is a dictionary t:emission, where it lists only the observed emissions.
    creates a custom dictionary object with a fallback value for queried keys not in the dictionary.
    serves as a drop-in replacement for current code.
    """

    class FallbackDict(dict):
        def __init__(self, default, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.default = default

        def __missing__(self, key):
            return self.default

    hmm = copy.deepcopy(hmm)  # protect again in place modification
    # Compute emissions weights for easier access
    emit_weights = FallbackDict(
        default=torch.ones(len(hmm.states)).type(dtype).to(device)
    )
    for t in obs_dict.keys():
        val = np.array([hmm.eprob[k, obs_dict[t]] for k in hmm.states])
        val = torch.from_numpy(val).type(dtype).to(device)
        emit_weights[t] = val

    return emit_weights


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


def single_simulation(hmm, min_time=0, stay=3, pro_before=10, ix_list=None):
    """
    Draws from hmm with addition constraint that we stay in each state for at least duration "stay"
    pro_before sets the maximum time horizon that promoter must occur by.
    """
    # Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, ix_list=ix_list, return_ix=True)
    init_prob, tmat, emat = hmm_params

    # Prepare dictionary for converting one_hot back to states
    state_ix, emit_ix = ix_list
    state_ix = {v: k for k, v in state_ix.items()}
    emit_ix = {v: k for k, v in emit_ix.items()}

    # Generate (X1,Y1)
    x_curr = random_draw(init_prob)
    current_state = state_ix[np.argmax(x_curr)]  # convert one-hot back to state
    x_list = [current_state]
    emit_dist = x_curr @ emat
    y_curr = random_draw(emit_dist)
    y_list = [emit_ix[np.argmax(y_curr)]]

    x_prev = x_curr

    # Initialize visit_trackers
    visit_pro = current_state == "pro"
    visit_dis = current_state == "dis"
    visit_enh = current_state == "enh"

    dis_visits = int(current_state == "dis")

    # Initialize state stay counter
    stay_counter = 1

    # Generate rest
    itr = 1  # iteration counter
    while current_state != "end":
        # By Markov property, just clamp to current stay until stay for required time
        if stay_counter < stay:
            stay_counter += 1
        else:
            # Transition to a new state
            x_curr = random_draw(x_prev @ tmat)
            if np.argmax(x_prev) != np.argmax(x_curr):
                stay_counter = 1  # Reset stay counter for the new state
                current_state = state_ix[np.argmax(x_curr)]
                emit_dist = x_curr @ emat
                x_prev = x_curr

                # Update visit_trackers
                visit_pro = visit_pro or current_state == "pro"
                visit_dis = visit_dis or current_state == "dis"
                visit_enh = visit_enh or current_state == "enh"

                if current_state == "dis":
                    # this condition already assumes transition to new state, so records new dis region.
                    dis_visits += 1

        # Constraints
        # check we hit promoter by pro_before

        itr += 1

        if itr == int(pro_before) and (not visit_pro):
            return False

        # pro < dis < enh

        if (not visit_pro) and visit_dis:
            return False

        if (not visit_dis) and visit_enh:
            return False

        y_curr = random_draw(emit_dist)

        x_list.append(current_state)
        y_list.append(emit_ix[np.argmax(y_curr)])

    # check if only one dis region
    if dis_visits != 1:
        return False

    if not visit_enh:
        return False

    return x_list, y_list


def simulation(hmm, min_time=0, stay=5, pro_before=30, ix_list=None, max_attempts=1000):
    """
    Repeatedly calls the simulation function until a valid full run is generated.
    Returns the first valid simulation (list of states and emissions).
    If no valid simulation is found within max_attempts, raises an exception.
    """
    for attempt in range(max_attempts):
        result = single_simulation(
            hmm, min_time, stay=stay, pro_before=pro_before, ix_list=ix_list
        )
        if result is not False:
            return result  # Return the valid simulation

    raise RuntimeError(
        f"Failed to generate a valid simulation after {max_attempts} attempts."
    )


# =====================================
# Inference
# =====================================


def Viterbi_torch_list(
    hmm,
    cst_list,
    obs,
    pro_before=30,
    dtype=torch.float32,
    device="cpu",
    debug=False,
    num_corr=0,
    hmm_params=None,
):
    """
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.
    """
    hmm = copy.deepcopy(hmm)  # protect again in place modification
    # Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    # Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(
        hmm, cst_list, dtype=dtype, device=device, return_ix=True, hmm_params=hmm_params
    )
    tmat, init_prob = hmm_params
    dims_list, init_ind_list, final_ind_list, ind_list = cst_params_list

    # Viterbi
    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)

    val = torch.empty((T, K) + tuple(dims_list), device="cpu")
    ix_tracker = torch.empty(
        (T, K) + tuple(dims_list), device="cpu"
    )  # will store flattened indices

    kr_indices = list(range(C + 1))
    kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]

    # Forward pass
    # V = torch.einsum('k,k,kr -> kr', init_prob, emit_weights[0], init_ind)

    V = torch.einsum(
        emit_weights[0], [0], init_prob, [0], *init_ind_list, kr_indices
    )  # (K,C1,C2,C3,...)
    V = V / (V.max() + num_corr)  # normalize for numerical stability
    val[0] = V.cpu()

    for t in range(1, T):
        # return kr_indices, ind_list, dims_list, C
        # V = torch.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = torch.einsum(
            val[t - 1].to(device),
            js_indices,
            tmat,
            [C + 1, 0],
            *ind_list,
            list(range(2 * C + 2)),
        )
        V = V.reshape(
            tuple(kr_shape) + (-1,)
        )  # colapse  the predecessor indices js into a single dim
        V = V / (V.max() + num_corr)
        max_ix = torch.argmax(V, axis=-1, keepdims=True)
        ix_tracker[t - 1] = max_ix.squeeze(-1)
        V = torch.take_along_dim(V, max_ix, axis=-1).squeeze(-1)
        # if t == T:
        #     # val[t] = torch.einsum('k,kr,kr -> kr',emit_weights[t],final_ind,V)
        #     val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices,*final_ind_list, kr_indices).cpu()
        # else:
        #     # val[t] = torch.einsum('k,kr -> kr', emit_weights[t],V)
        #     val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices, kr_indices).cpu()
        if t == pro_before:
            # Evaluate only the first constraint at time = 30
            val[t] = torch.einsum(
                emit_weights[t], [0], V, kr_indices, *final_ind_list[:2], kr_indices
            ).cpu()
        elif t == T - 1:
            # Evaluate all constraints at the last time
            val[t] = torch.einsum(
                emit_weights[t], [0], V, kr_indices, *final_ind_list[2:], kr_indices
            ).cpu()
        else:
            # Regular update without evaluating constraints
            val[t] = torch.einsum(emit_weights[t], [0], V, kr_indices, kr_indices).cpu()

    # return val
    state_ix = {v: k for k, v in state_ix.items()}
    # Backward pass
    opt_augstateix_list = []
    max_ix = int(torch.argmax(val[T - 1]).item())
    unravel_max_ix = np.unravel_index(max_ix, kr_shape)
    opt_augstateix_list = [np.array(unravel_max_ix).tolist()] + opt_augstateix_list

    ix_tracker = ix_tracker.reshape(T, -1)  # flatten again for easier indexing

    for t in range(T - 1):
        max_ix = int(ix_tracker[T - 2 - t, max_ix].item())
        unravel_max_ix = np.unravel_index(max_ix, kr_shape)
        opt_augstateix_list = [np.array(unravel_max_ix).tolist()] + opt_augstateix_list

    opt_state_list = [state_ix[k[0]] for k in opt_augstateix_list]
    if debug:
        return opt_state_list, opt_augstateix_list, val, ix_tracker
    return opt_state_list, opt_augstateix_list


# =====================================
# Sampling: Fixed and Variable Length
# =====================================


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


def ffbs_torch_list(
    hmm,
    cst_list,
    length_param,
    pro_before=30,
    dtype=torch.float32,
    device="cpu",
    debug=False,
    hmm_params=None,
):
    """
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    """
    hmm = copy.deepcopy(hmm)  # protect again in place modification
    # Generate emit_weights:
    if type(length_param) is int:
        emit_weights = np.ones(length_param)
        T = length_param
    elif isinstance(length_param, list):
        emit_weights = compute_emitweights(length_param, hmm)
        emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)
        T = emit_weights.shape[0]

    elif isinstance(length_param, tuple):
        T, emit_dict = length_param
        emit_weights = compute_emitweights_missing(
            emit_dict, hmm, dtype=dtype, device=device
        )

    else:
        raise ValueError(
            "length_param must be either an int, list of observations, or tuple containg length and dictionary of observed times"
        )

    # Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(
        hmm, cst_list, dtype=dtype, device=device, return_ix=True, hmm_params=hmm_params
    )
    tmat, init_prob = hmm_params
    dims_list, init_list, eval_list, upd_list = cst_params_list

    # #Assume that last constraint is the one whose sat time is estimated
    # #Parameters for the other fixed constraints
    # fixed_init, fixed_eval, fixed_upd = init_ind_list[:-2], final_ind_list[:-2], ind_list[:-2]

    # #Parameters for estimated sat time constraint.
    # sat_init, sat_true_eval, sat_upd = init_ind_list[-2:], final_ind_list[-2:], ind_list[-2:]
    # sat_false_eval = [1-sat_true_eval[0], sat_true_eval[1]]

    # Viterbi
    K = tmat.shape[0]
    C = len(dims_list)

    alpha = torch.empty((T, K) + tuple(dims_list), device="cpu")

    kr_indices = list(range(C + 1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]

    # initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    # indices are kurjvs
    alpha[0] = torch.einsum(
        emit_weights[0], [0], init_prob, [0], *init_list, kr_indices
    ).cpu()

    # Compute forward messages:
    for t in range(1, T):
        # Common terms, summing over js indices.
        V = torch.einsum(
            alpha[t - 1].to(device),
            js_indices,
            tmat,
            [C + 1, 0],
            emit_weights[t],
            [0],
            *upd_list,
            kr_indices,
        )

        V = (
            V / V.sum()
        )  # stepwise renormalization ok, as alpha dictates each steps sampling weights.

        if t == pro_before:
            alpha[t] = torch.einsum(V, kr_indices, *eval_list[:2], kr_indices).cpu()

        elif t == T - 1:
            alpha[t] = torch.einsum(V, kr_indices, *eval_list[2:], kr_indices).cpu()

        else:
            alpha[t] = V  # torch.einsum(V, kr_indices, kr_indices).cpu()

    # return alpha
    # Sample paths:
    upd_tensor_list = upd_list[::2]  # extract just the upd tensor, not the indices
    ix_list = [index_sampler(alpha[T - 1])]

    for t in range(T - 2, -1, -1):
        last_ix = ix_list[-1]

        transition_row_list = [tmat.cpu()[:, last_ix[0].item()], [0]]
        transition_row_list += list(
            chain.from_iterable(
                (upd.cpu()[last_ix[0].item(), last_ix[ix].item(), :], [ix])
                for ix, upd in enumerate(upd_tensor_list, start=1)
            )
        )

        probs = torch.einsum(*transition_row_list, kr_indices)
        probs = probs * alpha[t]
        ix_list.append(index_sampler(probs))

    # Decode.
    state_ix = {v: k for k, v in state_ix.items()}  # flip, so indices map to states

    sampled_path = [state_ix[s[0].item()] for s in ix_list]
    sampled_path.reverse()

    return sampled_path


def variable_length_sampling(
    hmm,
    time_cst,
    sample_cst,
    length_param,
    min_time=0,
    pro_before=30,
    dtype=torch.float32,
    device="cpu",
    debug=False,
    hmm_params=None,
):
    """
    Samples the stopping time, then samples a feasible path.

    time_cst is the constraint list used to sample stopping time: last constraint should be stopping time
    sample_cst is constraint list for sampling path.
    """
    probs, _, _ = satTime_torch_list(
        hmm,
        time_cst,
        length_param,
        min_time=min_time,
        pro_before=pro_before,
        dtype=dtype,
        device=device,
    )
    probs = probs / probs.sum()
    length = torch.multinomial(probs, num_samples=1).item() + min_time
    sample_path = ffbs_torch_list(
        hmm,
        sample_cst,
        length_param=(length, length_param[1]),
        pro_before=pro_before,
        dtype=dtype,
        device=device,
    )

    return sample_path, length


# =====================================
# Satisfaction Time
# =====================================
def satTime_torch_list(
    hmm,
    cst_list,
    length_param,
    min_time=0,
    pro_before=30,
    dtype=torch.float32,
    device="cpu",
    debug=False,
    hmm_params=None,
):
    """
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    """
    hmm = copy.deepcopy(hmm)  # protect again in place modification
    # Generate emit_weights:
    if type(length_param) is int:
        emit_weights = np.ones(length_param)
        T = length_param
    elif isinstance(length_param, list):
        emit_weights = compute_emitweights(length_param, hmm)
        emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)
        T = emit_weights.shape[0]

    elif isinstance(length_param, tuple):
        T, emit_dict = length_param
        emit_weights = compute_emitweights_missing(
            emit_dict, hmm, dtype=dtype, device=device
        )

    else:
        raise ValueError(
            "length_param must be either an int, list of observations, or tuple containg length and dictionary of observed times"
        )

    # Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(
        hmm, cst_list, dtype=dtype, device=device, return_ix=True, hmm_params=hmm_params
    )
    tmat, init_prob = hmm_params
    dims_list, init_ind_list, final_ind_list, ind_list = cst_params_list

    # Assume that last constraint is the one whose sat time is estimated
    # Parameters for the other fixed constraints
    fixed_init, fixed_eval, fixed_upd = (
        init_ind_list[:-2],
        final_ind_list[:-2],
        ind_list[:-2],
    )

    # Parameters for estimated sat time constraint.
    sat_init, sat_true_eval, sat_upd = (
        init_ind_list[-2:],
        final_ind_list[-2:],
        ind_list[-2:],
    )
    sat_false_eval = [1 - sat_true_eval[0], sat_true_eval[1]]

    # Viterbi
    K = tmat.shape[0]
    C = len(dims_list)

    alpha = torch.empty((T, K) + tuple(dims_list), device="cpu")
    gamma = torch.empty(alpha.shape, device="cpu")
    # beta doesn't need to track mediation space of estimate sat time. Dummy dim for last.
    beta = torch.empty((T, K) + tuple(dims_list[:-1]), device="cpu")
    sat_probs = torch.empty((T,))

    kr_indices = list(range(C + 1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]

    # initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    # indices are kurjvs
    gamma[0] = torch.einsum(
        emit_weights[0],
        [0],
        init_prob,
        [0],
        *fixed_init,
        *sat_init,
        *sat_false_eval,
        kr_indices,
    ).cpu()
    alpha[0] = torch.einsum(
        emit_weights[0],
        [0],
        init_prob,
        [0],
        *fixed_init,
        *sat_init,
        *sat_true_eval,
        kr_indices,
    ).cpu()
    beta[-1] = 1
    beta[-2] = torch.einsum(
        tmat,
        [C + 1, 0],
        *fixed_upd,
        *fixed_eval[2:],
        emit_weights[-1],
        [0],
        js_indices[:-1],
    ).cpu()

    fwd_norm = torch.zeros((T,))
    bck_norm = torch.zeros((T,))

    fwd_norm[0] = 1
    bck_norm[-1] = 1
    bck_norm[-2] = 1

    # the total normalization at alpha[t] is product of all up-to-present normalizations
    log_running_norm = 0
    # Compute forward messages:
    for t in range(1, T):
        # Common terms, summing over js indices.
        V = torch.einsum(
            gamma[t - 1].to(device),
            js_indices,
            tmat,
            [C + 1, 0],
            emit_weights[t],
            [0],
            *ind_list,
            kr_indices,
        )

        norm = V.sum()
        V = V / norm

        if norm.item() <= 0.0:
            raise ValueError(f"sums to 0! at time {t} on forward pass")

        log_running_norm += torch.log(norm)
        fwd_norm[t] = log_running_norm
        if torch.abs(log_running_norm) > 300:
            fwd_norm[: t + 1] = (
                fwd_norm[: t + 1] - log_running_norm
            )  # renormalize for stability
            log_running_norm = 0

        if t == pro_before:
            gamma[t] = torch.einsum(
                V, kr_indices, *sat_false_eval, *fixed_upd[:2], kr_indices
            ).cpu()
            alpha[t] = torch.einsum(
                V, kr_indices, *sat_true_eval, *fixed_upd[:2], kr_indices
            ).cpu()

        elif t == T - 1:
            gamma[t] = torch.einsum(
                V, kr_indices, *sat_false_eval, *fixed_eval[2:], kr_indices
            ).cpu()
            alpha[t] = torch.einsum(
                V, kr_indices, *sat_true_eval, *fixed_eval[2:], kr_indices
            ).cpu()

        else:
            gamma[t] = torch.einsum(V, kr_indices, *sat_false_eval, kr_indices).cpu()
            alpha[t] = torch.einsum(V, kr_indices, *sat_true_eval, kr_indices).cpu()

    # Compute backward messages
    log_running_norm = 0
    for t in range(T - 3, -1, -1):
        if t == pro_before:
            beta[t] = torch.einsum(
                beta[t + 1].to(device),
                kr_indices[:-1],
                tmat,
                [C + 1, 0],
                *fixed_upd,
                emit_weights[t + 1],
                [0],
                *fixed_eval[:2],
                js_indices[:-1],
            ).cpu()
        else:
            beta[t] = torch.einsum(
                beta[t + 1].to(device),
                kr_indices[:-1],
                tmat,
                [C + 1, 0],
                *fixed_upd,
                emit_weights[t + 1],
                [0],
                js_indices[:-1],
            ).cpu()
        norm = beta[t].sum()
        if norm.item() <= 0.0:
            raise ValueError(f"sums to 0! at time {t} on backward pass")
        beta[t] = beta[t] / norm
        # running_norm *= norm
        # bck_norm[t] = running_norm
        # if running_norm <= 1e-7:
        #     bck_norm[t:] = bck_norm[t:] / running_norm #renormalize for stability
        #     running_norm = 1

        log_running_norm += torch.log(norm)
        bck_norm[t] = log_running_norm
        if torch.abs(log_running_norm) > 300:
            bck_norm[t:] = bck_norm[t:] - log_running_norm  # renormalize for stability
            log_running_norm = 0

    # Compute moments
    for t in range(T):
        alpha_msg = alpha[t].to(device)
        beta_msg = beta[t].to(device)
        # sat_probs[t] = torch.einsum('...i,... ->', alpha_msg, beta_msg)
        sat_probs[t] = torch.einsum(
            alpha_msg, kr_indices, beta_msg, kr_indices[:-1], []
        )

    # Center the log normalization constants for stabilization
    fwd_norm = fwd_norm - fwd_norm.mean()
    bck_norm = bck_norm - bck_norm.mean()

    # return sat_probs, fwd_norm, bck_norm

    sat_probs = sat_probs / sat_probs.sum()  # normalize once for numerical stability
    sat_probs = sat_probs * torch.exp(fwd_norm + bck_norm)
    sat_probs = sat_probs[min_time:]
    sat_probs = sat_probs / sat_probs.sum()

    return sat_probs, alpha, beta


# =====================================
# Satisfaction Probability
# =====================================


def satprob_torch_list(
    hmm,
    cst_list,
    obs,
    pro_before=30,
    dtype=torch.float32,
    device="cpu",
    debug=False,
    num_corr=0,
    hmm_params=None,
):
    """
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_corr when normalizing.

    For DNA, always assume that the promoter constraint is first.

    Assume that last constraint is the one whose sat time to compute.
    """
    hmm = copy.deepcopy(hmm)  # protect again in place modification
    # Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm)
    emit_weights = torch.from_numpy(emit_weights).type(dtype).to(device)

    # Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(
        hmm, cst_list, dtype=dtype, device=device, return_ix=True, hmm_params=hmm_params
    )
    tmat, init_prob = hmm_params
    dims_list, init_list, eval_list, upd_list = cst_params_list

    # Assume that last constraint is the one whose sat time is estimated
    # Parameters for the other fixed constraints
    fixed_init, fixed_eval, fixed_upd = init_list[:-2], eval_list[:-2], upd_list[:-2]

    # Parameters for estimated sat time constraint.
    sat_init, sat_true_eval, sat_upd = init_list[-2:], eval_list[-2:], upd_list[-2:]
    sat_false_eval = [1 - sat_true_eval[0], sat_true_eval[1]]

    # Viterbi
    T = emit_weights.shape[0]
    K = tmat.shape[0]
    C = len(dims_list)

    kr_indices = list(range(C + 1))
    # fwd_kr_shape = (K,) + tuple(dims_list)
    js_indices = [k + C + 1 for k in kr_indices]

    # initialize. Let u,v denote the current/past indices of fixed constrained mediation space
    # indices are kurjvs
    alpha = torch.einsum(
        emit_weights[0], [0], init_prob, [0], *fixed_init, *sat_init, kr_indices
    )
    alpha = alpha / (alpha.sum())  # normalize for stability
    # Compute forward messages:
    for t in range(1, T - 1):
        # Common terms, summing over js indices.
        if t == pro_before:
            # alpha[t] = torch.einsum(V, kr_indices, *fixed_upd[:2], kr_indices).cpu()

            alpha = torch.einsum(
                alpha,
                js_indices,
                tmat,
                [C + 1, 0],
                emit_weights[t],
                [0],
                *upd_list,
                *fixed_eval[:2],
                kr_indices,
            )
        else:
            alpha = torch.einsum(
                alpha,
                js_indices,
                tmat,
                [C + 1, 0],
                emit_weights[t],
                [0],
                *upd_list,
                kr_indices,
            )

        alpha = alpha / (alpha.max() + num_corr)  # normalize for stability
    # compute final probs
    alpha_true = torch.einsum(
        alpha.to(device),
        js_indices,
        tmat,
        [C + 1, 0],
        emit_weights[T - 1],
        [0],
        *upd_list,
        *fixed_eval[2:],
        *sat_true_eval,
        kr_indices,
    )
    alpha_false = torch.einsum(
        alpha.to(device),
        js_indices,
        tmat,
        [C + 1, 0],
        emit_weights[T - 1],
        [0],
        *upd_list,
        *fixed_eval[2:],
        *sat_false_eval,
        kr_indices,
    )

    probs = torch.tensor(
        [alpha_true.sum().cpu().item(), alpha_false.sum().cpu().item()]
    )
    probs = probs / probs.sum()

    return probs  # , alpha_true, alpha_false
