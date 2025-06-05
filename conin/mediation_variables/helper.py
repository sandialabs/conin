import numpy as np
import json
from munch import Munch
import itertools
from collections import defaultdict
import random

def compute_emitweights(obs,hmm):
    '''
    Separately handles the computation of the 
    '''
    T = len(obs)
    K = len(hmm.states)
    #Compute emissions weights for easier access
    emit_weights = np.zeros((T,K))
    for t in range(T):
        emit_weights[t] = np.array([hmm.eprob[k,obs[t]] for k in hmm.states])

    return emit_weights

def arrayConvert(obs, hmm, cst, sat):
    '''
    Converts/generates relevant parameters/weights into numpy arrays for Baum-Welch.
    By assumption, the update/emission parameters associated with the constraint are static.
    For now, fix the emission probabilities.
    Only the hmm paramters are being optimized.
    '''
    #Initialize and convert all quantities  to np.arrays
    aux_space = list(itertools.product([True, False], repeat=cst.aux_size))
    T = len(obs)
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
    ind = np.zeros((M,K,M))
    init_ind = np.zeros((M,K))
    final_ind = np.zeros(M)

    for r in aux_space:
        final_ind[aux_ix[r]] = cst.cst_fun(r,sat)
        for i in hmm.states:
            init_ind[aux_ix[r],state_ix[i]] = cst.init_fun(i,r)
            for s in aux_space:
                ind[aux_ix[r],state_ix[i],aux_ix[s]] = cst.update_fun(r,i,s)
                
    cst_params = [init_ind,final_ind,ind]
    
    return hmm_params, cst_params 


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

def numpy2hmm(hmm_params, ix_list, tol = 1e-7, time_inhom = False):
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
    if time_inhom:
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
