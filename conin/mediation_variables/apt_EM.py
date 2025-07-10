import torch
import torch
from collections import defaultdict
import copy
import itertools
import time

def em_emitweights(apt_hmm, seq_list, mix_param = .81, dtype = torch.float32):
    hmm = copy.deepcopy(apt_hmm)
    E = len(hmm.emits)
    emit_ix = {s: i for i, s in enumerate(hmm.emits)}

    #Create the noisy emission matrix C.
    noisemat = mix_param*torch.eye(E) + (1-mix_param)/E*torch.ones((E,E))

    weight_list = []
    for seq in seq_list:
        T = len(seq)
        weight = torch.empty(T,E,dtype=dtype)
        for t in range(T):
            weight[t] = noisemat[:,emit_ix[seq[t]]] #P(O_t|Y = y), over all y
        weight_list.append(weight)
        
    return(weight_list)

def em_convertTensor(apt_hmm, cst_list, rand_init = True, dtype = torch.float32, device = 'cpu', return_ix = False):
    '''
    creates appropriate tensors for BW.
    if rand_init True, then will randomly initialize the transition, emissions matrix and delay parameter mu.

    IMPORTANT:
    Assumes states are ordered: [PRE, POST, OG, WAIT]
    Assumes emits are ordered: [None, Rest]
    '''
    #Initialize and convert all quantities  to np.arrays
    hmm = copy.deepcopy(apt_hmm)
    K = len(hmm.states)
    L = (K-2)//2 #number of OG states
    E = len(hmm.emits)
    
    state_ix = {s: i for i, s in enumerate(hmm.states)}
    emit_ix = {s: i for i, s in enumerate(hmm.emits)}

    if rand_init:
        #initialize mu
        mu = torch.rand(1).item()

        #intialize the original tmat and build in wait
        og_tmat = torch.rand((L,L+1),dtype=dtype).to(device) #OG to POST, OG
        og_tmat = og_tmat/og_tmat.sum(dim = 1,keepdims= True)
        tmat = torch.zeros((K,K), dtype=dtype ).to(device) #transition matrix
        tmat[1,1] = 1. #POST fixed to absorbing
        tmat[2:(2+L),1:(2+L)] = (1-mu)*og_tmat #OG > POST,OG
        tmat[(2+L):,1:(2+L)] = (1-mu)*og_tmat # WAIT > POST,OG
        tmat[0,0:(2+L)] = 1/(L+2) #intial PRE to uniformly over PRE,POST, OG
        tmat[2:(2+L),(2+L):] = mu*torch.eye(L).to(device) #OG > WAIT
        tmat[(2+L):,(2+L):] = mu*torch.eye(L).to(device) #WAIT > WAIT

        #initialize emissions
        emat = torch.rand((K, E), dtype=dtype ).to(device) #emissions matrix
        emat[2:(2+L),0] = 0 #OG states cannot emit None
        emat = emat/emat.sum(dim=1,keepdims=True)
        emat[(K-L):] = 0
        emat[(K-L):,0] = 1. #WAIT states always emit None
        emat[:2] = 0
        emat[:2,0] = 1. #PRE,POST always emit None
        init_prob = torch.zeros(K,dtype=dtype).to(device) #initial distribution
        for i in hmm.states:
            init_prob[state_ix[i]] = hmm.initprob[i]
    else:
        #Compute the hmm parameters
        tmat = torch.zeros((K,K), dtype=dtype ).to(device) #transition matrix
        emat = torch.zeros((K, E), dtype=dtype ).to(device) #emissions matrix
        init_prob = torch.zeros(K,dtype=dtype).to(device) #initial distribution
    
        for i in hmm.states:
            init_prob[state_ix[i]] = hmm.initprob[i]
            for j in hmm.states:
                tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]
            for e in hmm.emits:
                emat[state_ix[i],emit_ix[e]] = hmm.eprob[i,e]

        # noisemat = mix_param*torch.eye(E) + (1-mix_param)/E*torch.ones((E,E))
        # noisemat = noisemat.to(device)
        mu = hmm.mu
    hmm_params = [tmat, emat, init_prob, mu] #A,B,pi,mu in the note. 
    
    #Compute the cst parameters 
    init_ind_list = []
    ind_list = []
    dims_list = []
    cst_ix = 0
    C = len(cst_list)
    for cst in cst_list:
        cst = copy.deepcopy(cst)
        aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
        aux_ix = {s: i for i, s in enumerate(aux_space)}
        M = len(aux_space)
        ind = torch.zeros((K,K,E,M,M),dtype=dtype )
        init_ind = torch.zeros((K,E,M),dtype=dtype )
        # final_ind = torch.zeros((K,M),dtype=dtype ).to(device) no need. final ind is just 1 with compressed rep

                            
        for r in aux_space:
            for k in hmm.states:
                for e in hmm.emits:
                    curr_combined = (k,e)
                    init_ind[state_ix[k],emit_ix[e],aux_ix[r]] = cst.init_fun(curr_combined,r)
                    for s in aux_space:
                        for j in hmm.states:
                            #cst.update_fun only pulls first elt of the previous combined state.
                            prev_combined = (j,'None') 
                            ind[state_ix[j],state_ix[k], emit_ix[e], aux_ix[s],aux_ix[r]] = \
                            cst.update_fun(curr_combined,r,prev_combined,s)

        init_ind_list.append(init_ind)
        ind_list.append(ind)

    #now create I_{jkesr} by doing series of outerproducts and collapsing
    imat = ind_list[0]
    for ind in ind_list[1:]:
        #flatten (s,u) -> s' and (r,v) -> r'
        imat = torch.einsum('jkesr,jkeuv -> jkesurv',imat,ind) #outer product on last two dims
        imat = imat.reshape(*imat.shape[:-2],-1) #flatten rv into r'.
        imat = imat.reshape(K,K,E,-1,imat.size(-1)) #flateten su into s'

    #Care, the order of the rv indices must be same to match above.
    #If vr here but rv above, will not match.
    initmat = init_ind_list[0]
    for init_ind in init_ind_list[1:]:
        initmat = torch.einsum('ker,kev -> kerv',initmat,init_ind)
        initmat = initmat.reshape(K,E,-1) #flatten last two indices

    
        
                
    cst_params = [imat.to(device),initmat.to(device)]

    if return_ix:
        return hmm_params, cst_params, state_ix, emit_ix
    return hmm_params, cst_params 


def apt_BW(weight_list, hmm_params,cst_params, debug = False):
    '''
    Computes the summed moments (over batch, time, and intermediate variables)
    
    K, E, M are sizes of hidden, emit, and augmented respectively.
    All tensors already on GPU unless otherewise specified.
    We use the normalization procedure in Rabiner for numerical stability. See BW notes.
    
    IN:
    weight_list: list of CPU tensors of length B. each tensor has shape (T_i, E). 
    hmm_params: list of tensors: [A,B, pi, mu]. The transition, emission, intial, and delay parameter respecitvely. Initial dist is fixed to Pre.
    cst_params: list of tesnors: [ind,init_ind]
        ind_{jkesr} = P(M_t = r|Z_t = k, Y_t = e, Z_{t-1} = j, M_{t-1} = s) update function
        init_ind_{ker} = P(M_1 = r | Z_1 = k, Y_1 = e)
    debug: Boolean. if true, returns the computations of P(X,C) using each time point.

    OUT:
    ttl_gamma. gamma moments, summed over batch, time, and intnermediate variables.
    ttl_xi. xi moments. summed over same as above.
    debug_prob_list. if debug is True, then list of list. Each sublist is list of P(X,C) computed using different times.
    '''
    A, B, pi, mu = hmm_params #don't need to remember the og_mat
    ind, init_ind = cst_params
    K, E = B.shape
    M = ind.size(-1)
    device = A.device

    ttl_gamma = 0
    ttl_xi = 0
    log_prob = 0
    
    if debug:
        debug_prob_list = []


    for C_cpu in weight_list: #generate the messages for each sequence
        C = C_cpu.to(device)
        T = C.size(0)
        #First compute D_{jksr} out of loop, which will be reused several times
        D = torch.einsum('te,ke,jkesr -> tjksr', C, B, ind)
    
        #Create empty forward/backward messages. 
        alpha =  torch.empty((T,K,E,M)).to(device) #for now on GPU. see if run out of memory
        beta = torch.empty((T,K,M)).to(device)
        norms = torch.empty(T).to(device) #store the normalizing constants
        
        #Initialize alpha and beta
        message = torch.einsum('k,ke,e,ker -> ker', pi, B, C[0], init_ind)
        norm_message = 1/message.sum()
        norms[0] = norm_message #store normalizing constants
        alpha[0] = message*norm_message #normalize to 1
        
        
        
        #Forward messages
        for t in range(1,T): #last message no different since compressed formulation
            # ind2 = torch.einsum('js, jk, jkesr -> ker', past_alpha, A, ind) #intermediate product. manually split it up since not sure torch will do this.
            # message = torch.einsum('e,ke,ker -> ker', C[t], B, ind2)
            message = torch.einsum('e,ke,jds,jk,jkesr -> ker',C[t],B,alpha[t-1], A, ind)
            norm_message = 1/message.sum()
            norms[t] = norm_message
            alpha[t] = norm_message*message
        
        #Backward messages
        beta[T-1] = norms[T-1]
        
        for t in range(1,T):
            beta[T-1-t] = norms[T-1-t]*torch.einsum('js,kj,kjrs -> kr',beta[T-t], A, D[T-t])
            # beta[T-1-t] = norms[T-1-t]*torch.einsum('')
            
        
        #Compute the moments
        if debug: #dat prob P(X,C)
            all_dat_prob = torch.einsum('tker,tkr -> t', alpha, beta)
            all_dat_prob = all_dat_prob/norms #divide each by the appropriate normalizing constant
        #     dat_prob = all_dat_prob[0].item()
            debug_prob_list.append(all_dat_prob.cpu())
        # else:
        #     dat_prob = norms[0]*torch.einsum('ker,kr -> ', alpha[0], beta[0]).item()
        
        #We'll always sum both moments over time for A,B.
        #normalized message don't need to be divided by P(X,C)
        gamma = torch.einsum('tker,tkr ->  tke', alpha, beta)
        gamma = gamma/norms.view(T,1,1) #divide each gamme by appropriate norm constant
        if debug:
            og_gamma = gamma.clone()
            og_xi = torch.einsum('tjds,tkr,jk, tjksr -> t', alpha[:(T-1)],beta[1:],A, D[1:]) #xi time index starts at 2.
        gamma = gamma.sum(dim=0) #sum over time
        xi = torch.einsum('tjes,tkr,jk, tjksr -> jk', alpha[:(T-1)],beta[1:],A, D[1:]) #xi time index starts at 2.
        
        ttl_gamma += gamma
        ttl_xi += xi

        #finally, compute the log prob log P(X,C|\theta). See BW for formula derivation.
        log_prob += -1*torch.log(norms).sum().cpu().item() #divde by T so it doesn't get too large?

    if debug:
        return ttl_gamma, ttl_xi, log_prob, debug_prob_list, [og_gamma,og_xi]

    return ttl_gamma, ttl_xi, log_prob

def apt_preprocess(apt_hmm, device = 'cpu'):
    '''
    1. Rearranges the order of states to [PRE,POST,OG,WAIT]
    2. Computes the aggregation weights for the M step in wait_list
    '''
    hmm = copy.deepcopy(apt_hmm)
    hmm.states = ['PRE','POST'] + [s for s in hmm.states if s not in ['POST','PRE']]

    K = len(hmm.states)
    L = (K - 2)//2 #number of OG states

    wait_ix = torch.zeros((L,K)).to(device)
    delay_ix = torch.zeros((2,K,K)).to(device)

    #sum xi message for jk and j_w k. OG and its wait analogue
    for i in range(L):
        wait_ix[i,2+i] = 1 #OG
        wait_ix[i,2+i+L] = 1 #. WAIT. 

    delay_ix[0,2:,1:(2+L)] = 1 #transitions from OG+WAIT to POST+OG. The 1-mu terms
    
    for j in range(L): #the mu terms
        delay_ix[1,2+j,2+j+L] = 1 #OG to WAIT.
        delay_ix[1, 2+j+L, 2+j+L] = 1 #WAIT to WAIT

    return hmm, [wait_ix,delay_ix, L]
    

def apt_Mstep(hmm_params,wait_list, ttl_gamma, ttl_xi, fix_mu = None):
    '''
    Given the moments from the E-step, perform the M step.
    We don't update the initial dist pi, which is fixed to PRE.
    
    IMPORTANT. assume states are in order [PRE,POST, STATES, WAIT_STATES] where states  and wait analogues in same order.
    If | STATES | = L, then total 2 + 2L states in apt.
    
    IN:
    hmm_params: list of tensors: [A,B, pi, mu]. The transition, emission, intial, and delay parameter respecitvely. Initial dist is fixed to Pre.
    wait_list: list [wait_ix, delay_ix,  L].
        wait_ix: (L, K) tensor W. W_{ij} is 0,1 matrix for summing messages of STATES and their WAIT analogue. Omits POST and PRE.
        delay_ix: (2,K,K) tensor. W_{1jk} combined all messages that feed into C1. W_{2jk} for C2. See Baum-Welch Note
        L: integer. number of WAIT sattes.
    ttl_gamma: (K,E). Summed moments over batch, time, augmented var.
    ttl_xi: (J,K). Summed moments over same.

    OUT:
    new_hmm_params. same as above. we update the transition, emission, mu as above. Note that:
        - A: POST is fixed to be absorbing
        - B: PRE, POST. fixed to emit None.
    '''
    A_old , B_old, pi, _ = hmm_params
    wait_ix, delay_ix, L = wait_list

    device = A_old.device
    A = torch.zeros(A_old.shape).to(device)
    B = B_old.clone()
    
    #update the mu parameter
    mu_const = torch.einsum('cjk,jk -> c',delay_ix, ttl_xi)
    mu = 1/(1+(mu_const[0].item()/mu_const[1].item())) #mu^* = 1/(1+C1/C2)

    if fix_mu:
        mu = fix_mu #true APT delay
    #update the PRE > PRE,POST,OG transitions.
    A[0,:(2+L)] = ttl_xi[0,:(2+L)]/ttl_xi[0,:(2+L)].sum() #usual update eqn  for PRE

    #update STATE > POST, STATE transitions
    state_xi = torch.einsum('ij,jk -> ik', wait_ix, ttl_xi) #sum messages for OG and WAIT analogue. Transitions to POST, OG only
    ogA = state_xi[:,1:(2+L)]/state_xi[:,1:(2+L)].sum(dim = 1, keepdim=True)
    A[1,1] = 1. #POST absorbing
    A[2:(2+L), 1:(2+L)] = (1-mu)*ogA #fill in OG > POST, OG
    A[(2+L):, 1:(2+L)] = (1-mu)*ogA #copy WAIT > POST,OG

    #update transition to WAIT
    A[2:(2+L),(2+L):] = mu*torch.eye(L).to(device) #OG > WAIT
    A[(2+L):,(2+L):] = mu*torch.eye(L).to(device) #WAIT > WAIT

    #update emission matrix.
    #only update STATE emissions. PRE, POST, WAIT all fixed to None. OG cannot emit None
    B[2:(2+L),1:] = ttl_gamma[2:(2+L),1:]/ttl_gamma[2:(2+L),1:].sum(dim=1,keepdim=True)


    return [A, B,pi,mu]

def apt_Mstep_old(hmm_params,wait_list, ttl_gamma, ttl_xi):
    '''
    Given the moments from the E-step, perform the M step.
    We don't update the initial dist pi, which is fixed to PRE.
    
    IMPORTANT. assume states are in order [PRE,POST, STATES, WAIT_STATES] where states  and wait analogues in same order.
    If | STATES | = L, then total 2 + 2L states in apt.
    
    IN:
    hmm_params: list of tensors: [A,B, pi, mu]. The transition, emission, intial, and delay parameter respecitvely. Initial dist is fixed to Pre.
    wait_list: list [wait_ix, delay_ix,  L].
        wait_ix: (L, K) tensor W. W_{ij} is 0,1 matrix for summing messages of STATES and their WAIT analogue. Omits POST and PRE.
        delay_ix: (2,K,K) tensor. W_{1jk} combined all messages that feed into C1. W_{2jk} for C2. See Baum-Welch Note
        L: integer. number of WAIT sattes.
    ttl_gamma: (K,E). Summed moments over batch, time, augmented var.
    ttl_xi: (J,K). Summed moments over same.

    OUT:
    new_hmm_params. same as above. we update the transition, emission, mu as above. Note that:
        - A: POST is fixed to be absorbing
        - B: PRE, POST. fixed to emit None.
    '''
    A_old, B_old, pi, _ = hmm_params
    A = A_old.clone()
    B = B_old.clone()
    wait_ix, delay_ix, L = wait_list

    #update the PRE > PRE,POST,OG transitions.
    A[0,:(2+L)] = ttl_xi[0,:(2+L)]/ttl_xi[0,:(2+L)].sum() #usual update eqn  for PRE

    #update STATE > POST, STATE transitions
    state_xi = torch.einsum('ij,jk -> ik', wait_ix, ttl_xi) #sum messages for K  > K' and WAIT_K > K'
    ogA = state_xi/state_xi.sum(dim = 1, keepdim=True)
    A[2:(2+L), 1:(2+L)] = state_trans #fill in STATE > POST, STATE
    A[(2+L):, 1:(2+L)] = state_trans #copy WAIT_STATE > STATE

    #update emission matrix.
    #only update STATE emissions. PRE, POST, WAIT all fixed to None. OG cannot emit None
    B[2:(2+L),1:] = ttl_gamma[2:(2+L),1:]/ttl_gamma[2:(2+L),1:].sum(dim=1,keepdim=True)

    #update the mu parameter
    mu_const = torch.einsum('cjk,jk -> c',delay_ix, ttl_xi)
    mu = 1/(1+(mu_const[0].item()/mu_const[1].item())) #mu^* = 1/(1+C1/C2)

    return [A,state_trans, B,pi,mu]

def em_convertAPT(hmm_params, state_ix, emit_ix):
    '''
    Given a list of tensors, it converts the transition/emission matrix into dictionaries and returns them.
    '''
    tmat, emat, init_prob, mu = hmm_params
    tmat = tmat.cpu()
    emat = emat.cpu()
    
    tprob = defaultdict(int)
    eprob = defaultdict(int)

    states = list(state_ix.keys())
    emits = list(emit_ix.keys())
    for k in states:
        for j in states:
            tprob[j,k] = tmat[state_ix[j],state_ix[k]].item()
        for e in emits:
            eprob[k,e] = emat[state_ix[k],emit_ix[e]].item()

    return [tprob, eprob]

def apt_EM(apt_hmm, cst_list, seq_list,  device ='cpu', conv = 1e-10, max_step = 100, dtype = torch.float32, rand_init = True,\
           mix_param = .81, timer = False, fix_mu = None):
    '''
    Does EM on the APT, with constraints, with the following restrictions:
    1. Fix the intial distribution to pi('PRE') = 1
    2. Fix 'PRE','POST' and all WAIT states to emit 'None' with prob 1.
    3. OG states cannot emit None
    4. Cannot transition back to PRE once out of PRE.
    3. Enforce transitions out of a OG state and its WAIT analogue are the same. A_{jk} = A{j_w k}
    4. The delay parameter mu affects transitions from OG + WAIT to POST+OG by (1-mu). OG+ WAIT to WAIT by mu. PRE,POST have no WAIT.
    5. Rearranges the hidden state order of the APT to ['PRE','POST',OG, WAIT]
    6. The noisy emission (E,E) matrix is fixed, with noise dictate by mix_param. ie. We know how busy other processes are.

    IN: 
    apt_hmm. original apt object. 
    cst_list. list of constraint objects.
    seq_list. list of observation sequences.
    conv. numeric. convergence threshold. Stop when diff(A) + diff(B) < conv, where diff(x) = || x_new - x_old ||
        Tried doing log-prob, but for now seems to be no change. Guess is log-probs for long sequences way too small, espceially for a batch (ie. product of small).
    max_step. int. maximum number of steps
    rand_init. Boolean. if True, then will randomly initialize transition, emissions, and mu parameters.
    mix_param. numeric. set to P(Bob or Alice emit) = P((Bob and Alice don't emit)^c) = 1 - mu_bob * mu_alice

    OUT:
    new_apt. apt, with states reordered as above, with updated parameters.
    '''
    #Reorder the apt states
    ordered_apt, wait_list = apt_preprocess(apt_hmm, device = device)

    #Generate the hmm and cst params
    hmm_params, cst_params, state_ix, emit_ix = em_convertTensor(ordered_apt, cst_list, rand_init = rand_init, dtype = dtype, device = device, return_ix = True)

    #Generate the list of emit weights
    weight_list = em_emitweights(ordered_apt,seq_list, mix_param = mix_param, dtype = dtype)

    change = 999
    it = 0
    if timer:
        time_list = []
    while (it <= max_step) and (change > conv):
        start_time = time.time()
        ttl_gamma, ttl_xi, log_prob = apt_BW(weight_list, hmm_params,cst_params)
        run_time = time.time() - start_time
        if timer:
            time_list.append(run_time)
        new_hmm_params =  apt_Mstep(hmm_params,wait_list, ttl_gamma, ttl_xi, fix_mu = fix_mu)
        param_change = sum([torch.linalg.norm(old-new).cpu().item() for old,new in zip(hmm_params[:2],new_hmm_params[:2])])
        change = param_change + abs(new_hmm_params[3] - hmm_params[3])
        # if it % 10 == 0:
        print(f'Step:{it}, T+E change: {param_change}, Ttl change: {change}, neg log prob: {-1*log_prob}')
        it += 1
        hmm_params = new_hmm_params

    tprob, eprob = em_convertAPT(hmm_params, state_ix, emit_ix)

    ordered_apt.tprob = tprob
    ordered_apt.eprob = eprob
    ordered_apt.mu = hmm_params[3] #last is mu

    if timer:
        return ordered_apt, hmm_params, time_list
    return ordered_apt, hmm_params

def apt_EM_v2(apt_hmm, cst_list, seq_list,  device ='cpu', conv = 1e-10, max_step = 100, dtype = torch.float32, rand_init = True, \
              mix_param = .81, timer = False, fix_mu = None):
    '''
    Does EM on the APT, with constraints, with the following restrictions:
    1. Fix the intial distribution to pi('PRE') = 1
    2. Fix 'PRE','POST' and all WAIT states to emit 'None' with prob 1.
    3. OG states cannot emit None
    4. Cannot transition back to PRE once out of PRE.
    3. Enforce transitions out of a OG state and its WAIT analogue are the same. A_{jk} = A{j_w k}
    4. The delay parameter mu affects transitions from OG + WAIT to POST+OG by (1-mu). OG+ WAIT to WAIT by mu. PRE,POST have no WAIT.
    5. Rearranges the hidden state order of the APT to ['PRE','POST',OG, WAIT]
    6. The noisy emission (E,E) matrix is fixed, with noise dictate by mix_param. ie. We know how busy other processes are.

    IN: 
    apt_hmm. original apt object. 
    cst_list. list of constraint objects.
    seq_list. list of observation sequences.
    conv. numeric. convergence threshold. stop when abs(logprob(old - logprob(new))/logprob(old) < conv
        Tried doing log-prob, but for now seems to be no change. Guess is log-probs for long sequences way too small, espceially for a batch (ie. product of small).
    max_step. int. maximum number of steps
    rand_init. Boolean. if True, then will randomly initialize transition, emissions, and mu parameters.
    mix_param. numeric. set to P(Bob or Alice emit) = P((Bob and Alice don't emit)^c) = 1 - mu_bob * mu_alice

    OUT:
    new_apt. apt, with states reordered as above, with updated parameters.
    '''
    #Reorder the apt states
    ordered_apt, wait_list = apt_preprocess(apt_hmm, device = device)

    #Generate the hmm and cst params
    hmm_params, cst_params, state_ix, emit_ix = em_convertTensor(ordered_apt, cst_list, rand_init = rand_init, dtype = dtype, device = device, return_ix = True)

    #Generate the list of emit weights
    weight_list = em_emitweights(ordered_apt,seq_list, mix_param = mix_param, dtype = dtype)

    change = 999
    it = 0

    start_time = time.time()
    ttl_gamma, ttl_xi, log_prob = apt_BW(weight_list, hmm_params,cst_params)
    run_time = time.time() - start_time
    if timer:
        time_list = [run_time]
    while (it <= max_step) and (change > conv):
        new_hmm_params =  apt_Mstep(hmm_params,wait_list, ttl_gamma, ttl_xi, fix_mu = fix_mu)
        start_time = time.time()
        ttl_gamma, ttl_xi, new_log_prob = apt_BW(weight_list, new_hmm_params,cst_params)
        run_time = time.time() - start_time
        if timer:
            time_list.append(run_time)
            
        change = abs((new_log_prob - log_prob)/log_prob)
        # if it % 10 == 0:
        print(f'Step:{it}, neg log prob: {-1*new_log_prob}, log_prob_change %: {change}')
        it += 1
        hmm_params = new_hmm_params
        log_prob = new_log_prob

    tprob, eprob = em_convertAPT(hmm_params, state_ix, emit_ix)

    ordered_apt.tprob = tprob
    ordered_apt.eprob = eprob
    ordered_apt.mu = hmm_params[3] #last is mu

    if timer:
        return ordered_apt, hmm_params, time_list
    return ordered_apt, hmm_params