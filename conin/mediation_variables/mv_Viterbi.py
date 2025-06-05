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
                        curr_val = val[t-1,j,s]*hmm.tprob[j,k]*cst.update_fun(r,j,s)
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
                        curr_val = val[t-1,j,s]*hmm.tprob[j,k]*cst.update_fun(r,j,s)
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

    return(opt_augstate, opt_state)

