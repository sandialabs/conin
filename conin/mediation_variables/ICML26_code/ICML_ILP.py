import torch
import pulp
import time
import numpy as np


from typing import Callable, Optional, Tuple, List, Tuple


######################################################################
# ILP
######################################################################


def hmm_constrained_inference(
    observations: torch.Tensor,
    transition: torch.Tensor,
    emission: torch.Tensor, 
    initial: torch.Tensor,
    N: int
) -> Tuple[torch.Tensor, float]:
    """
    Perform exact constrained inference in an HMM using ILP.
    Enforces that the subsequence [0, 1, 2, ..., N-1] occurs somewhere in the output.
    
    Args:
        observations: Tensor of shape (T,) with observed symbols
        transition: Tensor of shape (K, K) with log transition probabilities
        emission: Tensor of shape (K, M) with log emission probabilities
        initial: Tensor of shape (K,) with log initial state probabilities
        N: Length of required subsequence [0, 1, ..., N-1]
    
    Returns:
        Tuple of (state_sequence, solve_time):
            state_sequence: Tensor of shape (T,) with the most likely state sequence
            solve_time: Time in seconds to solve the ILP
    """
    T = observations.shape[0]
    K = transition.shape[0]  # number of states
    
    # Convert to numpy and ensure we're working with log probabilities
    trans = transition.detach().cpu().numpy()
    emit = emission.detach().cpu().numpy()
    init = initial.detach().cpu().numpy()
    obs = observations.detach().cpu().numpy()
    
    # Create ILP model
    model = pulp.LpProblem("HMM_Constrained_Inference", pulp.LpMaximize)
    
    # Binary variables: z[t, k] = 1 if state k is active at time t
    z = {}
    for t in range(T):
        for k in range(K):
            z[t, k] = pulp.LpVariable(f"z_{t}_{k}", cat='Binary')
    
    # Objective: maximize log probability
    obj = []
    
    # Initial state contribution
    for k in range(K):
        obj.append(init[k] * z[0, k])
    
    # Emission probabilities
    for t in range(T):
        for k in range(K):
            obj.append(emit[k, int(obs[t])] * z[t, k])
    
    # Transition probabilities
    trans_vars = {}
    for t in range(T - 1):
        for k1 in range(K):
            for k2 in range(K):
                # Create binary variable for transition from k1 to k2 at time t
                trans_var = pulp.LpVariable(f"trans_{t}_{k1}_{k2}", cat='Binary')
                trans_vars[t, k1, k2] = trans_var
                
                # trans_var = 1 iff z[t, k1] = 1 AND z[t+1, k2] = 1
                model += trans_var <= z[t, k1]
                model += trans_var <= z[t + 1, k2]
                model += trans_var >= z[t, k1] + z[t + 1, k2] - 1
                
                obj.append(trans[k1, k2] * trans_var)
    
    model += pulp.lpSum(obj)
    
    # Constraint: exactly one state active at each time
    for t in range(T):
        model += pulp.lpSum([z[t, k] for k in range(K)]) == 1
    
    # Add subsequence constraint: [0, 1, 2, ..., N-1] must occur somewhere
    if N > 0:
        # Check if sequence is feasible
        if N > T:
            raise ValueError(f"Subsequence length N={N} cannot be longer than sequence length T={T}")
        if N > K:
            raise ValueError(f"Subsequence length N={N} cannot be longer than number of states K={K}")
        
        # Create binary variables indicating if the subsequence starts at position t
        subseq_starts = {}
        
        for t in range(T - N + 1):  # Valid starting positions
            subseq_starts[t] = pulp.LpVariable(f"subseq_start_{t}", cat='Binary')
            
            # subseq_starts[t] = 1 iff the full subsequence [0,1,...,N-1] occurs starting at t
            # This means: z[t, 0] = 1 AND z[t+1, 1] = 1 AND ... AND z[t+N-1, N-1] = 1
            for i in range(N):
                model += subseq_starts[t] <= z[t + i, i]
            
            # If all positions match, subseq_starts[t] can be 1
            model += subseq_starts[t] >= pulp.lpSum([z[t + i, i] for i in range(N)]) - N + 1
        
        # At least one occurrence of the subsequence
        model += pulp.lpSum([subseq_starts[t] for t in range(T - N + 1)]) >= 1
    
    # Solve
    start_time = time.time()
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time
    
    if model.status != pulp.LpStatusOptimal:
        raise ValueError(f"Optimization failed with status {pulp.LpStatus[model.status]}")
    
    # Extract solution
    solution = np.zeros(T, dtype=np.int64)
    for t in range(T):
        for k in range(K):
            if pulp.value(z[t, k]) > 0.5:  # Binary variable is 1
                solution[t] = k
                break
    
    return torch.tensor(solution, dtype=torch.long), solve_time