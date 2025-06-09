import itertools

def Viterbi(obs, hmm):
    '''
    Just regular Viterbi
    
    obs: list of observed emissions
    hmm: Munch object containig our hmm object    
    '''
    val = {} #initialize value dictionary

    for k in hmm.states:
            val[0,k] = hmm.initprob[k]*hmm.eprob[k,obs[0]]
            
    ix_tracker = {} #this is used in the backwards step to identify the optimal sequence
    
    #Forward: compute value function and generate index
    for t in range(1,len(obs)):
        for k in hmm.states:
            max_val = -1 # set to dummy variable. will do brute-force search for max
            argmax = None #initialize argmax for ix_tracker
            for j in hmm.states:
                curr_val = val[t-1,j]*hmm.tprob[j,k]
                if curr_val > max_val:
                    max_val = curr_val
                    argmax = j
            val[t,k] = max_val*hmm.eprob[k,obs[t]]
            ix_tracker[t-1,k] = argmax
    
    #Backward: compute the values of the optimal sequence
    max_val = -1
    best_state = None
    for k in hmm.states:
        curr_val = val[len(obs) - 1,k]
        if curr_val > max_val:
            max_val = curr_val
            best_state = k
    opt_state = [best_state]
    
    for t in range(len(obs) -1):
        best_state = ix_tracker[len(obs) - 2 -t,best_state]
        opt_state = [best_state] + opt_state

    return opt_state


def mv_Viterbi(obs, hmm, cst = None, sat = True):
    '''
    Does Viterbii with intermediate variables. In this version, the constraint is included as an binary emission at the last time. 
    This formulation allows us to easily with inference in the case where the constraint is satisfied or not.
    
    obs: list of observed emissions
    hmm: Munch object containig our hmm object
    cst: Munch object containing our constraint (cst) object
    sat. Boolean determining whether the constraint is ture or not
    
    '''
    if cst is None:
        return Viterbi(obs,hmm)
        
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size)) #create list of auxillary state to iterate over
    val = {} #initialize value dictionary

    for k in hmm.states:
        for r in aux_space:
            val[0,k,r] = cst.init_fun(k,r)*hmm.initprob[k]*hmm.eprob[k,obs[0]]
            
    ix_tracker = {} #this is used in the backwards step to identify the optimal sequence
    
    #Forward: compute value function and generate index
    for t in range(1,len(obs)):
        for k in hmm.states:
            for r in aux_space:
                max_val = -1 # set to dummy variable. will do brute-force search for max
                argmax = None #initialize argmax for ix_tracker
                for j in hmm.states:
                    for s in aux_space:
                        curr_val = val[t-1,j,s]*hmm.tprob[j,k]*cst.update_fun(r,k,s) #j to k
                        if curr_val > max_val:
                            max_val = curr_val
                            argmax = (j,s)
                if t == (len(obs)-1): #ie. at the last time we add in the constraint
                    val[t,k,r] = max_val*hmm.eprob[k,obs[t]]*cst.cst_fun(r,sat)
                else:
                    val[t,k,r] = max_val*hmm.eprob[k,obs[t]]
                ix_tracker[t-1,k,r] = argmax
    
    #Backward: compute the values of the optimal sequence
    max_val = -1
    best_state = None
    for k in hmm.states:
        for r in aux_space:
            curr_val = val[len(obs) - 1,k,r]
            if curr_val > max_val:
                max_val = curr_val
                best_state = (k,r)
    opt_augstate = [best_state]            
    opt_state = [best_state[0]]
    
    for t in range(len(obs) -1):
        best_state = ix_tracker[len(obs) - 2 -t,best_state[0], best_state[1]]
        opt_augstate = [best_state] + opt_augstate #append at the front
        opt_state = [best_state[0]] + opt_state

    return(opt_augstate, opt_state)

def mv_Viterbi_v2(obs, hmm, cst = None, sat = True):
    '''
    Does Viterbii with intermediate variables. In this version, the constraint is included as an binary emission at the last time. 
    This formulation allows us to easily with inference in the case where the constraint is satisfied or not.
    
    obs: list of observed emissions
    hmm: Munch object containig our hmm object
    cst: Munch object containing our constraint (cst) object
    sat. Boolean determining whether the constraint is ture or not
    
    '''
    if cst is None:
        return Viterbi(obs,hmm)
        
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size)) #create list of auxillary state to iterate over
    val = {} #initialize value dictionary

    for k in hmm.states:
        for r in aux_space:
            val[0,k,r] = cst.init_fun(k,r)*hmm.initprob[k]*hmm.eprob[k,obs[0]]
            
    ix_tracker = {} #this is used in the backwards step to identify the optimal sequence
    
    #Forward: compute value function and generate index
    for t in range(1,len(obs)):
        print(t)
        for k in hmm.states:
            for r in aux_space:
                max_val = -1 # set to dummy variable. will do brute-force search for max
                argmax = None #initialize argmax for ix_tracker
                for j in hmm.states:
                    for s in aux_space:
                        curr_val = val[t-1,j,s]*hmm.tprob[j,k]*cst.update_fun(k,r,j,s)
                        if curr_val > max_val:
                            max_val = curr_val
                            argmax = (j,s)
                if t == (len(obs)-1): #ie. at the last time we add in the constraint
                    val[t,k,r] = max_val*hmm.eprob[k,obs[t]]*cst.cst_fun(k,r,sat)
                else:
                    val[t,k,r] = max_val*hmm.eprob[k,obs[t]]
                ix_tracker[t-1,k,r] = argmax
    
    #Backward: compute the values of the optimal sequence
    max_val = -1
    best_state = None
    for k in hmm.states:
        for r in aux_space:
            curr_val = val[len(obs) - 1,k,r]
            if curr_val > max_val:
                max_val = curr_val
                best_state = (k,r)
    opt_augstate = [best_state]            
    opt_state = [best_state[0]]
    
    for t in range(len(obs) -1):
        best_state = ix_tracker[len(obs) - 2 -t,best_state[0], best_state[1]]
        opt_augstate = [best_state] + opt_augstate #append at the front
        opt_state = [best_state[0]] + opt_state

    return opt_augstate, opt_state


def mv_Viterbi_time(obs, hmm, cst, sat = True):
    '''
    Version where the emission probabiltiies are allowed to be time-inhomgenous.
    
    Does Viterbii with intermediate variables. In this version, the constraint is included as an binary emission at the last time. 
    This formulation allows us to easily with inference in the case where the constraint is satisfied or not.
    
    obs: list of observed emissions
    hmm: Munch object containig our hmm object
    cst: Munch object containing our constraint (cst) object
    sat. Boolean determining whether the constraint is ture or not
    
    '''
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size)) #create list of auxillary state to iterate over
    val = {} #initialize value dictionary

    for k in hmm.states:
        for r in aux_space:
            val[0,k,r] = cst.init_fun(k,r)*hmm.initprob[k]*hmm.eprob[0,k,obs[0]]
            
    ix_tracker = {} #this is used in the backwards step to identify the optimal sequence
    
    #Forward: compute value function and generate index
    for t in range(1,len(obs)):
        for k in hmm.states:
            for r in aux_space:
                max_val = -1 # set to dummy variable. will do brute-force search for max
                argmax = None #initialize argmax for ix_tracker
                for j in hmm.states:
                    for s in aux_space:
                        curr_val = val[t-1,j,s]*hmm.tprob[j,k]*cst.update_fun(k,r,j,s)
                        if curr_val > max_val:
                            max_val = curr_val
                            argmax = (j,s)
                if t == (len(obs)-1): #ie. at the last time we add in the constraint
                    val[t,k,r] = max_val*hmm.eprob[t,k,obs[t]]*cst.cst_fun(r,sat)
                else:
                    val[t,k,r] = max_val*hmm.eprob[t,k,obs[t]]
                ix_tracker[t-1,k,r] = argmax
    
    #Backward: compute the values of the optimal sequence
    max_val = -1
    best_state = None
    for k in hmm.states:
        for r in aux_space:
            curr_val = val[len(obs) - 1,k,r]
            if curr_val > max_val:
                max_val = curr_val
                best_state = (k,r)
    opt_augstate = [best_state]            
    opt_state = [best_state[0]]
    
    for t in range(len(obs) -1):
        best_state = ix_tracker[len(obs) - 2 -t,best_state[0], best_state[1]]
        opt_augstate = [best_state] + opt_augstate #append at the front
        opt_state = [best_state[0]] + opt_state

    return opt_augstate, opt_state 

def mv_Viterbi_numpy(hmm_params, cst_params, emit_weights):
    '''
    numpy version. hmm_params, cst_params are list of numpy arrays
    '''
    opt_augstateix_list = []
    
    tmat, init_prob = hmm_params
    init_ind,final_ind,ind = cst_params
    
    T = emit_weights.shape[0]
    K, M = init_ind.shape

    val = np.empty((T,K,M))
    ix_tracker = np.empty((T,K,M)) #will store flattened indices

    #Forward pass
    V = np.einsum('k,k,kr -> kr', init_prob, emit_weights[0], init_ind)
    val[0] = V
    for t in range(1,T):
        print(t)
        V = np.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = V.reshape((K,M,-1))
        max_ix = np.argmax(V, axis = -1, keepdims = True)
        ix_tracker[t-1] = max_ix.squeeze()
        V = np.take_along_axis(V, max_ix, axis=-1).squeeze()
        if t == T:
            val[t] = np.einsum('k,kr,kr -> kr',emit_weights[t],final_ind,V)
        else:
            val[t] = np.einsum('k,kr -> kr', emit_weights[t],V)
        

    #Backward pass

    #Initialize the last index
    max_ix = int(np.argmax(val[T-1]).item())
    max_k, max_r =divmod(max_ix, M)
    opt_augstateix_list.append((max_k,max_r))

    ix_tracker = ix_tracker.reshape(T,-1) #flatten again for easier indexing    
    
    for t in range(T-1):
        max_ix =  int(ix_tracker[T-2-t,max_ix].item())
        max_k, max_r = divmod(max_ix, M)
        opt_augstateix_list.append((max_k,max_r))

    return opt_augstateix_list

def mv_Viterbi_torch(hmm_params, cst_params, emit_weights):
    '''
    torch, GPU-accelerated version. hmm_params, cst_params are list of torch tensors
    '''
    opt_augstateix_list = []
    
    tmat, init_prob = hmm_params
    init_ind,final_ind,ind = cst_params
    
    T = emit_weights.shape[0]
    K, M = init_ind.shape

    val = torch.empty((T,K,M), device = tmat.device)
    ix_tracker = torch.empty((T,K,M), device = tmat.device) #will store flattened indices

    #Forward pass
    V = torch.einsum('k,k,kr -> kr', init_prob, emit_weights[0], init_ind)
    val[0] = V
    for t in range(1,T):
        print(t)
        V = torch.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = V.reshape((K,M,-1))
        max_ix = torch.argmax(V, axis = -1, keepdims = True)
        ix_tracker[t-1] = max_ix.squeeze()
        V = torch.take_along_dim(V, max_ix, axis=-1).squeeze()
        if t == T:
            val[t] = torch.einsum('k,kr,kr -> kr',emit_weights[t],final_ind,V)
        else:
            val[t] = torch.einsum('k,kr -> kr', emit_weights[t],V)
        

    #Backward pass

    #Initialize the last index
    max_ix = int(np.argmax(val[T-1]).item())
    max_k, max_r = divmod(max_ix, M)
    opt_augstateix_list.append((max_k,max_r))

    ix_tracker = ix_tracker.reshape(T,-1) #flatten again for easier indexing    
    
    for t in range(T-1):
        max_ix =  int(ix_tracker[T-2-t,max_ix].item())
        max_k, max_r = divmod(max_ix, M)
        opt_augstateix_list.append((max_k,max_r))

    return opt_augstateix_list


