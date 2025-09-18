import numpy as np
import torch
import json
from munch import Munch
import itertools
from collections import defaultdict
import random
import copy

def process_load(names_list, folder_path = '', delay = None):
    '''
    Build the hmm object.
    Also, reading in the json converts tuples to string. need to convert back to tuples.
    delay is a list of mu's, the probabiltiy of entering a waiting state.
    '''
    if delay:
        assert len(delay) == len(names_list)
    process_list = []
    it = 0
    for names in names_list:
        file_path = folder_path + f'{names}.json'
        with open(file_path,'r') as file:
            process = json.load(file)

        if delay:
            mu = delay[it]
            it += 1
            if names.startswith('apt'): #no waiting for PRE,POST.
                tprob = {eval(k): round((1-mu)*v,5) for k,v in process['transition_probs'].items() if eval(k)[0] not in ['PRE','POST']}
                add_prob = {eval(k): v for k,v in process['transition_probs'].items() if eval(k)[0] in ['PRE','POST']}
                tprob.update(add_prob)
            else:
                tprob = {eval(k): round((1-mu)*v,5) for k,v in process['transition_probs'].items()}
        else:
            tprob = {eval(k): v for k,v in process['transition_probs'].items()}

        #Augment the user emissions so they're also of the form (Server, (Server,Action))
        if names.startswith('apt'):
            eprob = {eval(k): v for k,v in process['emission_probs'].items()}
        else:
            eprob = {}
            for k, v in process['emission_probs'].items():
                key = eval(k)
                if key[1] is None:
                    eprob[key[0],None] = v
                else:
                    eprob[(key[0],key)] = v
        
        states = set()
        emits = set()
        
        for k in tprob.keys():
            states.update(k)
        for k in eprob.keys():
            emits.add(k[1])

        states = list(states)
        
        if delay:
            for k in list(states): #need to create a separate copy since we're appending to states
                if k in ['PRE','POST']: #no waiting for PRE, POST.
                    continue
                wait_state = f'WAIT_{k}'
                tprob[k,wait_state] = mu
                for j in states:
                    if (k,j) in tprob.keys():
                        tprob[wait_state,j] = tprob[k,j] # 1 - mu factor already applied
                        tprob[wait_state,wait_state] = mu
                eprob[wait_state,None] = 1.
                states.append(wait_state)
            emits.add(None)

        emits = list(emits)

        #Convert everything to defaultdict, where referencing a non-existent key returns a 0.
        tprob = defaultdict(int, tprob)
        eprob = defaultdict(int, eprob)
        initprob = defaultdict(int, process['start_probs'])
        
        
        process_hmm = Munch(name = names, states = states, emits = emits, tprob = tprob, \
                           eprob = eprob, initprob = initprob)

        if delay:
            process_hmm.mu = mu


        process_list.append(process_hmm)

    return process_list

def hmm2numpy(hmm, ix_list = None, return_ix = False, emit_inhom = False):
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

    if emit_inhom:
        T = len(emat)
        emat = np.zeros((T,K,M))
    else:
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
    if emit_inhom:
        for t in range(T):
            for i in hmm.states:
                for m in hmm.emits:
                    if (i,m) not in hmm.eprob:
                        continue
                    emat[t,state_ix[i],emit_ix[m]] = hmm.eprob[i,m]
    else:
        for i in hmm.states:
            for m in hmm.emits:
                if (i,m) not in hmm.eprob:
                    continue
                emat[state_ix[i],emit_ix[m]] = hmm.eprob[i,m]

    hmm_params = [init_prob, tmat, emat]

    if return_ix:
        return hmm_params, [state_ix, emit_ix] 
    return hmm_params

def numpy2hmm(hmm_params, ix_list, tol = 1e-7, emit_inhom = False):
    '''
    If time_inhom is true, then emat is assumed to be a list of matrices.
    '''
    state_ix, emit_ix = ix_list
    init_prob, tmat, emat = hmm_params
    initprob = defaultdict(int)
    tprob = defaultdict(int)
    eprob = defaultdict(int)
    K, M = len(state_ix), len(emit_ix)

    #reverse the dicts, so indices map to states
    state_ix = {v:k for k,v in state_ix.items()}
    emit_ix = {v:k for k,v in emit_ix.items()}
    #initprob
    for i in range(K):
        val = init_prob[i].item()
        if abs(val) > tol:
            initprob[state_ix[i]] = val

    for i in range(K):
        for j in range(K):
            val = tmat[i,j].item()
            if abs(val) > tol:
                tprob[state_ix[i],state_ix[j]] = val

    #eprob
    if emit_inhom:
        for t in range(len(emat)):
            for i in range(K):
                for j in range(M):
                    val = emat[t][i,j].item()
                    if abs(val) > tol:
                        eprob[t,state_ix[i],emit_ix[j]] = val
    else:
        for i in range(K):
            for j in range(M):
                val = emat[i,j].item()
                if abs(val) > tol:
                    eprob[state_ix[i],emit_ix[j]] = val

    #Convert everything to defaultdict, where referencing a non-existent key returns a 0.
    # tprob = defaultdict(int, tprob)
    # eprob = defaultdict(int, eprob)
    # initprob = defaultdict(int, initprob)
                
    return initprob, tprob, eprob

def forward_marginals(hmm_params, length):
    '''
    Given an hmm, computes the sequence of marginal hidden-emission probabilities.
    IN
    hmm_params:
    1. k x k matrix of hidden transiions
    2. k x n emission matrix. 
    3. k start probabilities
    All are numpy arrays. Row is start, col is end.

    ix_list: list of state and emits dictionaries that map all hidden/emits to indices.
    
    
    '''
    init_prob, tmat, emat = hmm_params
    hidden_marginal = init_prob
    emit_marginal = [hidden_marginal @ emat]
    for t in range(1,length):
        hidden_marginal = hidden_marginal @ tmat
        emit_marginal.append(hidden_marginal @ emat)
        
    return emit_marginal

def random_draw(p):
    '''
    p is a 1D np array. 
    single random draw from probability vector p and encode as 1-hot.
    '''
    n = len(p)
    p = p/p.sum()
    draw = np.random.choice(n,p=p)
    one_hot = np.zeros(n, dtype = int)
    one_hot[draw] = 1
    
    return one_hot


def simulation(hmm,time, ix_list = None, emit_inhom = False):
    '''
    generates a full run for specified time.
    homogenous is True if the emission probs are time-homogenous.
    transition matrix aways assumed to be homogenous
    '''
    #Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, ix_list = ix_list, return_ix = True, emit_inhom = emit_inhom) 
    init_prob, tmat, emat = hmm_params

    #Prepare dictionary for converting one_hot back to states
    state_ix, emit_ix = ix_list
    state_ix = {v:k for k,v in state_ix.items()}
    emit_ix = {v:k for k,v in emit_ix.items()}
    
    #Generate (X1,Y1)
    x_prev = random_draw(init_prob)
    x_list = [state_ix[np.argmax(x_prev)]] #convert one-hot back to state
    if emit_inhom:
        y_curr = random_draw(x_prev @ emat[0])
    else:
        y_curr = random_draw(x_prev @ emat)
    y_list = [emit_ix[np.argmax(y_curr)]]

    #Generate rest
    for t in range(1,time):
        x_curr = random_draw(x_prev @ tmat)
        if emit_inhom:
            y_curr = random_draw(x_curr @ emat[t])
        else:
            y_curr = random_draw(x_curr @ emat)
        x_list.append(state_ix[np.argmax(x_curr)])
        y_list.append(emit_ix[np.argmax(y_curr)])
        x_prev = x_curr

    return x_list, y_list

def simulation_knowledge(hmm, cst_list, ix_list = None, emit_inhom = False):
    '''
    for the apt, generates a run that stops whenever the "POST" state is encountered.
    '''
    #Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, ix_list = ix_list, return_ix = True, emit_inhom = emit_inhom) 
    init_prob, tmat, emat = hmm_params
    
    #Create dictionaries for generating mask for transitions/emissions
    state_ix, emit_ix = ix_list
    K, M = len(state_ix), len(emit_ix)
    
    tmat_mask_dict = {}
    eprob_mask_dict = {}
    for cst in cst_list:
        t_mask = np.ones((K,K))
        e_mask = np.ones((K,M))
        for ft in cst.forbidden_transitions:
            t_mask[state_ix[ft[0]],state_ix[ft[1]]] = 0
        for fe in cst.forbidden_emissions:
            e_mask[state_ix[fe[0]],emit_ix[fe[1]]] = 0
        tmat_mask_dict[cst.knowledge_state] = t_mask
        eprob_mask_dict[cst.knowledge_state] = e_mask
    
    state_ix = {v:k for k,v in state_ix.items()}
    emit_ix = {v:k for k,v in emit_ix.items()}
    
    notyet_knowledge = list(tmat_mask_dict.keys())  
    
    tmat_curr = tmat * np.prod(list(tmat_mask_dict.values()), axis = 0)
    emat_curr = emat * np.prod(list(eprob_mask_dict.values()), axis = 0)
    
    x_prev = random_draw(init_prob)
    x_state = state_ix[np.argmax(x_prev)] #convert one-hot back to state
    x_list = [x_state] 
    if emit_inhom:
        y_curr = random_draw(x_prev @ emat_curr[0])
    else:
        y_curr = random_draw(x_prev @ emat_curr)
    y_state = emit_ix[np.argmax(y_curr)]
    y_list = [y_state]

    #Generate rest
    while x_state != 'POST':
        x_curr = random_draw(x_prev @ tmat_curr)
        if emit_inhom:
            y_curr = random_draw(x_curr @ emat_curr[t])
        else:
            y_curr = random_draw(x_curr @ emat_curr)
        x_state = state_ix[np.argmax(x_curr)]
        y_state = emit_ix[np.argmax(y_curr)]
        hid_emit = (x_state,y_state) 
        if hid_emit in notyet_knowledge: #if knowledge state, gets rid of it from the mask
            tmat_mask_dict.pop(hid_emit)
            eprob_mask_dict.pop(hid_emit)
            tmat_curr = tmat * np.prod(list(tmat_mask_dict.values()), axis = 0)
            emat_curr = emat * np.prod(list(eprob_mask_dict.values()), axis = 0)
            notyet_knowledge = list(tmat_mask_dict.keys())
            
        x_list.append(x_state)
        y_list.append(y_state)
        x_prev = x_curr

    return x_list, y_list

def simulation_apt(hmm, ix_list = None, emit_inhom = False):
    '''
    for the apt, generates a run that stops whenever the "POST" state is encountered.
    '''
    #Get numpy version of hmm parameters
    hmm_params, ix_list = hmm2numpy(hmm, ix_list = ix_list, return_ix = True, emit_inhom = emit_inhom) 
    init_prob, tmat, emat = hmm_params

    #Prepare dictionary for converting one_hot back to states
    state_ix, emit_ix = ix_list
    state_ix = {v:k for k,v in state_ix.items()}
    emit_ix = {v:k for k,v in emit_ix.items()}
    
    #Generate (X1,Y1)
    x_prev = random_draw(init_prob)
    x_state = state_ix[np.argmax(x_prev)] #convert one-hot back to state
    x_list = [x_state] 
    if emit_inhom:
        y_curr = random_draw(x_prev @ emat[0])
    else:
        y_curr = random_draw(x_prev @ emat)
    y_list = [emit_ix[np.argmax(y_curr)]]

    #Generate rest
    while x_state != 'POST':
        x_curr = random_draw(x_prev @ tmat)
        if emit_inhom:
            y_curr = random_draw(x_curr @ emat[t])
        else:
            y_curr = random_draw(x_curr @ emat)
        x_state = state_ix[np.argmax(x_curr)]
        x_list.append(x_state)
        y_list.append(emit_ix[np.argmax(y_curr)])
        x_prev = x_curr

    return x_list, y_list

def combined_simulation(apt_hmm, user_list, cst_list = None):
    '''
    assume all processes are time homogenous
    '''
    if cst_list:
        apt_hidden, apt_emit = simulation_knowledge(apt_hmm,cst_list)
    else:
        apt_hidden, apt_emit = simulation_apt(apt_hmm)
    T = len(apt_emit)
    total_emits = [apt_emit]
    for user in user_list:
        total_emits.append(simulation(user, T)[1]) #only record the emission of each user 

    combined_emits = []
    for t in range(T):
        valid_emits = [emits[t] for emits in total_emits if emits[t] is not None] #exclude the None emits at time t
        if len(valid_emits) == 0:
            random_emit = None
        else:
            random_emit = random.choice(valid_emits)
        combined_emits.append(random_emit)
    return [apt_hidden, apt_emit], combined_emits

def check_valid(x_list, y_list, cst_list, return_knowledge = False):
    '''
    Checks if the sequence of hidden-emits satisfy the constraints
    Simulation is assumed to be a list of tuples.
    '''
    ft_dict = {c.knowledge_state:c.forbidden_transitions for c in cst_list}
    fe_dict = {c.knowledge_state:c.forbidden_emissions for c in cst_list}
    state_times = {c.knowledge_state:1e10 for c in cst_list} #initialize to some large time
    notyet_knowledge = list(ft_dict.keys())
    attained_states = set()
    
    ft = sum(list(ft_dict.values()),[])
    fe = sum(list(fe_dict.values()),[])

    T = len(x_list) #Will always be Pre:None.
    x_prev = x_list[0]
    valid = True
    for t in range(1,T):
        x_curr = x_list[t]
        y_curr = y_list[t]
        hid_emit = (x_curr,y_curr)
        if hid_emit in fe:
            if not return_knowledge:
                return False
            valid = False
            
        if (x_prev,x_curr) in ft:
            if not return_knowledge:
                return False
            valid = False

        if  hid_emit in notyet_knowledge:
            ft_dict.pop(hid_emit)
            fe_dict.pop(hid_emit)
            state_times[hid_emit] = t
            attained_states.add(hid_emit)
            notyet_knowledge = list(ft_dict.keys())
            ft = sum(list(ft_dict.values()),[])
            fe = sum(list(fe_dict.values()),[])
            
        x_prev = x_curr
    if return_knowledge:
        return valid, attained_states, list(state_times.values()) #should be in same order as in cst_list
    return True

def check_within_distance(x, y, d):
    # Ensure both lists are of equal length
    if len(x) != len(y):
        raise ValueError("Both lists must be of equal length.")
    
    # Create a list of boolean values based on the condition
    result = [abs(x_i - y_i) <= d for x_i, y_i in zip(x, y)]
    
    return result


def hmm2numpy_apt(hmm, ix_list = None, return_ix = False):
    '''
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    '''
    #Initialize and convert all quantities  to np.arrays
    state_ix = {s: i for i, s in enumerate(hmm.states)}

    if ix_list:
        emit_ix = ix_list[1]
    else:
        emit_ix = {s: i for i, s in enumerate(hmm.emits)}


    K, M = len(state_ix), len(emit_ix)
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

def create_combined_ix(apt_hmm, user_list):
    '''
    Generate a combined index dictionary for all possible servers and actions.
    '''
    combined_servers = set()
    combined_emits = set(apt_hmm.emits)
    
    for user in user_list:
        combined_servers.update(set(user.states))
        combined_emits.update(set(user.emits))
        
    #Generate the combined indices
    combined_server_ix = {s:i for i,s in enumerate(list(combined_servers))}
    combined_emits_ix = {s:i for i,s in enumerate(list(combined_emits))}

    return [combined_server_ix,combined_emits_ix]

def lapt_mixture(apt_hmm, user_list, length, mix_weights = None, return_ix = False):
    '''
    IN 
    apt_hmm: Munch object of the apt
    user_list: list of user hmms.
    length: int. how long we run all 3 processes.
    mix_weights: optional argument to supply np.array of mixture weights. If empty, then by default will just do uniform mixture.
        The weight for the original apt emat will be 1 - sum(mix_weights)
    add_delay: Boolean on whether the delay should be incorporated in the model.
    '''
    apt = copy.deepcopy(apt_hmm) #deepcopy to avoid funkiness.
    
    #Get unified indexing for all servers and emits
    usr_state_ix, emit_ix = create_combined_ix(apt, user_list)

    #Convert dicts to numpy arrays
    apt_params, apt_ix_list = hmm2numpy_apt(apt, ix_list = [usr_state_ix, emit_ix], return_ix = True)
    apt_state_ix = apt_ix_list[0]
    user_params = []
    for user in user_list:
        user_params.append(hmm2numpy(user, ix_list = [usr_state_ix, emit_ix]))

    #Compute the marginals over time
    user_marg_list = []
    for params in user_params:
        user_marg_list.append(forward_marginals(params, length))
    

    #Generate the weights
    n = len(user_marg_list)

    if mix_weights is None:
        mix_weights = np.array(1/(n+1)).repeat(n+1)
    emat_w = 1 - mix_weights.sum()

    mix_emat = []

    init_prob, tmat, emat = apt_params
    
    for t in range(length):
        curr_emat = emat_w*emat
        for i in range(n):
            curr_emat += mix_weights[i]*user_marg_list[i][t]
        curr_emat = np.around(curr_emat, decimals = 6)
        mix_emat.append(curr_emat)

    initprob, tprob, eprob = numpy2hmm([init_prob, tmat, mix_emat], [apt_state_ix, emit_ix], tol = 1e-7, emit_inhom = True)

    apt.eprob = eprob
    apt.eprob_time = length
    if return_ix:
        return apt, [apt_state_ix, emit_ix]
    return apt

def tier_mixture(apt_hmm, user_list, length, mix_weights = None, return_ix = False):
    '''
    IN 
    apt_hmm: Munch object of the apt
    user_list: list of user hmms.
    length: int. how long we run all 3 processes.
    mix_weights: optional argument to supply np.array of mixture weights. If empty, then by default will just do uniform mixture.
        The weight for the original apt emat will be 1 - sum(mix_weights)
    add_delay: Boolean on whether the delay should be incorporated in the model.
    '''
    apt_hmm = apt_hmm.copy() #copy so later in-place methods don't overwrite another variable named apt_hmm.
    
    #Get unified indexing for all servers and emits
    usr_state_ix, emit_ix = create_combined_ix(apt_hmm, user_list)

    #Convert dicts to numpy arrays
    apt_params, apt_ix_list = hmm2numpy_apt(apt_hmm, ix_list = [usr_state_ix, emit_ix], return_ix = True)
    apt_state_ix = apt_ix_list[0]
    user_params = []
    for user in user_list:
        user_params.append(hmm2numpy(user, ix_list = [usr_state_ix, emit_ix]))

    #Compute the marginals over time
    user_marg_list = []
    for params in user_params:
        user_marg_list.append(forward_marginals(params, length))
    

    #Generate the weights
    n = len(user_marg_list)

    if mix_weights is None:
        mix_weights = np.array(1/(n+1)).repeat(n+1)
    emat_w = 1 - mix_weights.sum()

    mix_emat = []

    init_prob, tmat, emat = apt_params
    
    for t in range(length):
        curr_emat = emat_w*emat
        for i in range(n):
            curr_emat += mix_weights[i]*user_marg_list[i][t]
        
        mix_emat.append(curr_emat)

    _, _, eprob = numpy2hmm([init_prob, tmat, mix_emat], [apt_state_ix, emit_ix], tol = 1e-7, emit_inhom = True)

    #Make sure everything is default dict.
    apt_hmm.eprob = defaultdict(int,eprob)
    apt_hmm.tprob = defaultdict(int,apt_hmm.tprob)
    apt_hmm.initprob = defaultdict(int,apt_hmm.tprob)
    apt_hmm.eprob_time = length
    if return_ix:
        return apt_hmm, [apt_state_ix, emit_ix]
    return apt_hmm



def create_noisy_apt(apt_hmm, mix_param, tol = 1e-7):
    '''
    Original APT: X -> Y

    Tiered APT: (X,Y) -> hat{Y}

    
    '''
    apt = copy.deepcopy(apt_hmm) #deepcopy since there's still some funkiness going on.
    M = len(apt.emits) #number of hidden states
    
    #Creat noisy emissions matrix
    eprob = defaultdict(int)
    #For now, create a noiseless emission, where the emission of the APT is the observed emission
    for k in apt.states:
        for e in apt.emits:
            eprob[k,e] = mix_param*apt.eprob[k,e] + (1- mix_param)/M 
        
    new_apt = Munch(name = apt.name, states = apt.states, emits = apt.emits, tprob = apt.tprob, \
                       eprob = eprob, initprob = apt.initprob)
    
    if apt.mu:
        new_apt.mu = apt.mu

    return new_apt

def create_tiered_apt(apt_hmm, tol = 1e-7):
    '''
    Original APT: X -> Y

    Tiered APT: (X,Y) -> hat{Y}

    
    '''
    apt = copy.deepcopy(apt_hmm) #deepcopy since there's still some funkiness going on.
    new_states = list(apt.eprob.keys())
    
    #Create transition matrix
    tprob = defaultdict(int)
    #dicts should be default dict, so referencing non-existent key returns 0.
    for state1 in new_states:
        for state2 in new_states:
            h1, e1 = state1
            h2, e2 = state2
            val = apt.tprob[h1,h2]*apt.eprob[h2,e2] #P(X_t, Y_t | X_{t-1}, Y_{t-1}) = P(Y_t|X_t)P(X_t | X_{t-1})
            if abs(val) > 1e-7: #only record the non-zero transitions
                tprob[state1,state2] = round(val,6)
    
    #Creat emission matrix
    eprob = defaultdict(int)
    #For now, create a noiseless emission, where the emission of the APT is the observed emission
    for state in new_states:
        h, e = state
        eprob[state,e] = 1.
    
    #Create initial distribution
    initprob = defaultdict(int)
    for state in new_states:
        h, e = state
        val = apt.initprob[h]*apt.eprob[h,e]
        if abs(val) > 1e-7:
            initprob[state] = round(val,6)
    
    new_apt = Munch(name = apt.name, states = new_states, emits = apt.emits, tprob = tprob, \
                       eprob = eprob, initprob = initprob)
    
    if apt.mu:
        new_apt.mu = apt.mu

    return new_apt


def arrayConvert(hmm, cst, sat, device = None):
    '''
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    '''
    #Initialize and convert all quantities  to np.arrays
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
    K = len(hmm.states)
    M = len(aux_space)
    
    state_ix = {s: i for i, s in enumerate(hmm.states)}
    aux_ix = {s: i for i, s in enumerate(aux_space)}

    #Compute the hmm parameters
    tmat = np.zeros((K,K))
    init_prob = np.zeros(K)

    for i in hmm.states:
        init_prob[state_ix[i]] = hmm.initprob[i]
        for j in hmm.states:
            tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]

    hmm_params = [tmat, init_prob]
    
    #Compute the cst parameters    
    ind = np.zeros((K,M,K,M))
    init_ind = np.zeros((K,M))
    final_ind = np.zeros((K,M))

    for r in aux_space:
        for k in hmm.states:
            final_ind[state_ix[k], aux_ix[r]] = cst.cst_fun(k,r,sat)
            init_ind[state_ix[k],aux_ix[r]] = cst.init_fun(k,r)
            for s in aux_space:
                for j in hmm.states:
                    ind[state_ix[k],aux_ix[r],state_ix[j],aux_ix[s]] = cst.update_fun(k,r,j,s)
                
    cst_params = [init_ind,final_ind,ind]

    if device:
        hmm_params = [torch.from_numpy(param).to(device) for param in hmm_params]
        cst_params = [torch.from_numpy(param).to(device) for param in cst_params]

    return hmm_params, cst_params 

def compute_emitweights(obs,hmm, time_hom = True):
    '''
    Separately handles the computation of the 
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    T = len(obs)
    K = len(hmm.states)
    #Compute emissions weights for easier access
    emit_weights = np.zeros((T,K))
    for t in range(T):
        if time_hom:
            emit_weights[t] = np.array([hmm.eprob[k,obs[t]] for k in hmm.states])
        else:
            emit_weights[t] = np.array([hmm.eprob[t,k,obs[t]] for k in hmm.states])
    return emit_weights


def numpy2tensor(hmm_params, cst_params, emit_weights, device):
    '''
    Converts all the numpy arrays to torch tensors
    '''
    hmm_params_torch = [torch.from_numpy(array).to(device) for array in hmm_params]
    cst_params_torch = [torch.from_numpy(array).to(device) for array in cst_params]
    emit_weights_torch = torch.from_numpy(emit_weights).to(device)

    return hmm_params_torch, emit_weights_torch, emit_weights_torch

def convertTensor_list(hmm, cst_list, sat, dtype = torch.float16, device = 'cpu', return_ix = False):
    '''
    cst_list is a list of the individual csts.
    '''
    #Initialize and convert all quantities  to np.arrays
    hmm = copy.deepcopy(hmm)
    K = len(hmm.states)
    assert len(cst_list) == len(sat)
    
    state_ix = {s: i for i, s in enumerate(hmm.states)}

    #Compute the hmm parameters
    tmat = torch.zeros((K,K), dtype=dtype ).to(device)
    init_prob = torch.zeros(K, dtype=dtype ).to(device)

    for i in hmm.states:
        init_prob[state_ix[i]] = hmm.initprob[i]
        for j in hmm.states:
            tmat[state_ix[i],state_ix[j]] = hmm.tprob[i,j]

    hmm_params = [tmat, init_prob]
    
    #Compute the cst parameters 
    init_ind_list = []
    final_ind_list = []
    ind_list = []
    dims_list = []
    cst_ix = 0
    C = len(cst_list)
    for cst in cst_list:
        cst = copy.deepcopy(cst)
        aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
        aux_ix = {s: i for i, s in enumerate(aux_space)}
        M = len(aux_space)
        ind = torch.zeros((K,M,K,M),dtype=dtype ).to(device)
        init_ind = torch.zeros((K,M),dtype=dtype ).to(device)
        final_ind = torch.zeros((K,M),dtype=dtype ).to(device)
    
        for r in aux_space:
            for k in hmm.states:
                final_ind[state_ix[k], aux_ix[r]] = cst.cst_fun(k,r,sat[cst_ix])
                init_ind[state_ix[k],aux_ix[r]] = cst.init_fun(k,r)
                for s in aux_space:
                    for j in hmm.states:
                        ind[state_ix[k],aux_ix[r],state_ix[j],aux_ix[s]] = cst.update_fun(k,r,j,s)

        #indices are [0 = k,  (1 dim for each cst r_i = i + 1)  0 <= i <= n - 1 
        # init_ind_list.append((init_ind,[0,cst_ix + 1]))
        # final_ind_list.append((final_ind, [0, cst_ix + 1]))
        # #indices are [0 = k,(1 dim for each cst r_i = i + 1), n + 1 = j, (1 dim for s_i = i+n+2)] 
        # ind_list.append((ind, [0, cst_ix + 1, C + 1, cst_ix + C + 2]))
        # dims_list.append(M)

        init_ind_list += [init_ind,[0,cst_ix + 1]]
        final_ind_list += [final_ind, [0, cst_ix + 1]]
        #indices are [0 = k,(1 dim for each cst r_i = i + 1), n + 1 = j, (1 dim for s_i = i+n+2)] 
        ind_list += [ind, [0, cst_ix + 1, C + 1, cst_ix + C + 2]]
        dims_list.append(M)
        cst_ix += 1
                
    cst_params = [dims_list, init_ind_list,final_ind_list,ind_list]

    if return_ix:
        return hmm_params, cst_params, state_ix
    return hmm_params, cst_params 

def Viterbi_torch_list(hmm, cst_list, obs, sat,  time_hom = True, dtype = torch.float16,  device = 'cpu', debug = False, num_cst = 0):
    '''
    more optimized torch implementation of Viterbi. The constraint all evolve independently (ie. factorial), so no need to create a big U_krjs matrix. Instead, just multiply along given dim. Still require computing V_{krjs}, but this should help.
    For numerica underflow, we normalize the value at each time. Also, we add a small constant num_cst when normalizing.
    '''
    hmm = copy.deepcopy(hmm) #protect again in place modification
    #Generate emit_weights:
    emit_weights = compute_emitweights(obs, hmm, time_hom)
    emit_weights = torch.from_numpy(emit_weights).type(torch.float16).to(device)

    #Generate hmm,cst params:
    hmm_params, cst_params_list, state_ix = convertTensor_list(hmm,cst_list, sat, dtype = dtype, \
                                                               device = device, return_ix = True)   
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
    V = V/(V.max() + num_cst) #normalize for numerical stability
    val[0] = V.cpu()
    for t in range(1,T):
        # V = torch.einsum('js,jk,krjs -> krjs',val[t-1],tmat,ind)
        V = torch.einsum(val[t-1].to(device), js_indices, tmat, [C+1,0], *ind_list, list(range(2*C + 2)))
        V = V.reshape(tuple(kr_shape) + (-1,))
        V = V/(V.max() + num_cst)
        max_ix = torch.argmax(V, axis = -1, keepdims = True)
        ix_tracker[t-1] = max_ix.squeeze()
        V = torch.take_along_dim(V, max_ix, axis=-1).squeeze()
        if t == T:
            # val[t] = torch.einsum('k,kr,kr -> kr',emit_weights[t],final_ind,V)
            val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices,*final_ind_list, kr_indices).cpu()
        else:
            # val[t] = torch.einsum('k,kr -> kr', emit_weights[t],V)
            val[t] = torch.einsum(emit_weights[t],[0], V, kr_indices, kr_indices).cpu()
        
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


