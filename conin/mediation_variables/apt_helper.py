import numpy as np
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
    initprob = {}
    tprob = {}
    eprob = {}
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
    tprob = defaultdict(int, tprob)
    eprob = defaultdict(int, eprob)
    initprob = defaultdict(int, initprob)
                
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

def simulation(hmm,time, ix_list = None, emit_inhom = False):
    '''
    generates a full run for specified time.
    homogenous is True if the emission probs are time-homogenous.
    transition matrix aways assumed to be homogenous
    '''
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

def lapt_heir(apt_hmm, user_list, length, mix_weights = None, return_ix = False):
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

    initprob, tprob, eprob = numpy2hmm([init_prob, tmat, mix_emat], [apt_state_ix, emit_ix], tol = 1e-7, emit_inhom = True)

    apt_hmm.eprob = eprob
    apt_hmm.eprob_time = length
    if return_ix:
        return apt_hmm, [apt_state_ix, emit_ix]
    return apt_hmm


def combined_simulation(apt_hmm, user_list, time):
    '''
    assume all processes are time homogenous
    '''
    apt_hidden, apt_emit = simulation(apt_hmm, time)
    
    total_emits = [apt_emit]
    for user in user_list:
        total_emits.append(simulation(user, time)[1]) #only record the emission of each user 

    combined_emits = []
    for t in range(time):
        valid_emits = [emits[t] for emits in total_emits if emits[t] is not None] #exclude the None emits at time t
        if len(valid_emits) == 0:
            random_emit = None
        else:
            random_emit = random.choice(valid_emits)
        combined_emits.append(random_emit)
    return [apt_hidden, apt_emit], combined_emits


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
