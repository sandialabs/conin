import torch
import itertools
import random
import copy
import time
from typing import Callable, Optional, Tuple, List, Tuple


######################################################################
# Beam Search with Incentive Functions
######################################################################


def hmm_beam_search(
    transition_matrix: torch.Tensor,  # A: (N, N) - transition probabilities
    emission_matrix: torch.Tensor,    # B: (N, M) - emission probabilities
    initial_vector: torch.Tensor,     # π: (N,) - initial state probabilities
    observations: torch.Tensor,       # O: (T,) - sequence of observations
    beam_width: int = 5,
    bonus_fn: Optional[Callable[[int, int, int], float]] = None
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    Beam search for HMM decoding with optional bonus function.
    
    Args:
        transition_matrix: Transition probabilities (N x N) - will be converted to log space
        emission_matrix: Emission probabilities (N x M) - will be converted to log space
        initial_vector: Initial state probabilities (N,) - will be converted to log space
        observations: Sequence of observation indices (T,)
        beam_width: Number of hypotheses to keep at each step
        bonus_fn: Optional bonus function(prev_state, next_state, time_step) -> bonus_score
        
    Returns:
        paths: List of top-k state sequences
        scores: Scores for each path
    """
    N = transition_matrix.shape[0]  # Number of states
    T = observations.shape[0]       # Sequence length
    device = transition_matrix.device
    
    # Convert to log space for numerical stability
    log_transition_matrix = torch.log(transition_matrix)
    log_emission_matrix = torch.log(emission_matrix)
    log_initial_vector = torch.log(initial_vector)
    
    # Initialize beam
    beam = [(log_initial_vector[s].item(), [s]) for s in range(N)]
    beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]
    
    # Process each observation
    for t in range(T):
        obs = observations[t].item()
        candidates = []
        
        for score, path in beam:
            prev_state = path[-1]
            
            for next_state in range(N):
                transition_score = log_transition_matrix[prev_state, next_state].item()
                emission_score = log_emission_matrix[next_state, obs].item()
                
                bonus = 0.0
                if bonus_fn is not None:
                    bonus = bonus_fn(prev_state, next_state, t)
                
                new_score = score + transition_score + emission_score + bonus
                new_path = path + [next_state]
                
                candidates.append((new_score, new_path))
        
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    
    paths = [path for _, path in beam]
    scores = torch.tensor([score for score, _ in beam], device=device)
    
    return paths, scores

def hmm_beam_search_vectorized(
    transition_matrix: torch.Tensor,  # A: (N, N)
    emission_matrix: torch.Tensor,    # B: (N, M)
    initial_vector: torch.Tensor,     # π: (N,)
    observations: torch.Tensor,       # O: (T,)
    beam_width: int = 5,
    bonus_fn: Optional[Callable[[int, int, int], float]] = None,
    device: torch.device = torch.device('cpu'), 
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    Vectorized beam search for HMM decoding.
    """
    N = transition_matrix.shape[0]
    T = observations.shape[0]
    
    # Convert to log space
    log_transition_matrix = torch.log(transition_matrix + 1e-10).to(device)
    log_emission_matrix = torch.log(emission_matrix + 1e-10).to(device)
    log_initial_vector = torch.log(initial_vector + 1e-10).to(device)
    
    # Initialize beam: get top beam_width initial states
    top_initial_scores, top_initial_states = torch.topk(log_initial_vector, 
                                                         min(beam_width, N))
    beam_scores = top_initial_scores  # (beam_width,)
    beam_paths = [[s.item()] for s in top_initial_states]
    
    # Process each observation
    for t in range(T):
        obs = observations[t].item()
        current_beam_size = len(beam_paths)
        
        # Vectorized computation for all beam × state combinations
        # Shape: (current_beam_size,)
        prev_states = torch.tensor([path[-1] for path in beam_paths], 
                                   device=device, dtype=torch.long)
        
        # Get transition scores: (current_beam_size, N)
        transition_scores = log_transition_matrix[prev_states, :]
        
        # Get emission scores: (N,) -> broadcast to (current_beam_size, N)
        emission_scores = log_emission_matrix[:, obs].unsqueeze(0).expand(current_beam_size, -1)
        
        # Compute base scores: (current_beam_size, N)
        # beam_scores is (current_beam_size,), unsqueeze to (current_beam_size, 1) for broadcasting
        base_scores = beam_scores.unsqueeze(1) + transition_scores + emission_scores
        
        # Apply bonus function if provided
        if bonus_fn is not None:
            if hasattr(bonus_fn, 'bonus_matrix'):
                # Vectorized bonus lookup: (current_beam_size, N)
                bonus_scores = bonus_fn.bonus_matrix[prev_states, :]
                base_scores = base_scores + bonus_scores
            else:
                # Fallback to loop-based approach
                bonus_matrix = torch.zeros((current_beam_size, N), device=device)
                for i, path in enumerate(beam_paths):
                    prev_state = path[-1]
                    for next_state in range(N):
                        bonus_matrix[i, next_state] = bonus_fn(prev_state, next_state, t)
                base_scores = base_scores + bonus_matrix
        
        # Flatten and get top-k
        flat_scores = base_scores.flatten()  # (current_beam_size * N,)
        top_k_scores, top_k_indices = torch.topk(flat_scores, 
                                                  min(beam_width, flat_scores.shape[0]))
        
        # Decode indices back to (beam_idx, state_idx)
        beam_indices = top_k_indices // N
        state_indices = top_k_indices % N
        
        # Update beam
        new_beam_paths = []
        for beam_idx, state_idx in zip(beam_indices.cpu().numpy(), 
                                       state_indices.cpu().numpy()):
            new_path = beam_paths[beam_idx] + [state_idx]
            new_beam_paths.append(new_path)
        
        beam_paths = new_beam_paths
        beam_scores = top_k_scores
    
    return beam_paths, beam_scores

def create_sequence_bonus_incentive(
    constraint_length: int,
    bonus_per_transition: float = 2.0
) -> Callable[[int, int, int], float]:
    """
    Creates an incentive function that encourages the consecutive sequence [0,1,2,...,N-1].
    Gives bonuses for transitions (n, n+1) where n < constraint_length - 1.
    
    Args:
        constraint_length: Length of the consecutive sequence to encourage (e.g., 5 for [0,1,2,3,4])
        bonus_per_transition: Bonus score added for each (n, n+1) transition
        
    Returns:
        Bonus function with signature (prev_state, next_state, time_step) -> bonus_score
    """
    
    # Generate required transitions: (0,1), (1,2), ..., (N-2, N-1)
    required_transitions = set((i, i+1) for i in range(constraint_length - 1))
    
    def bonus_fn(prev_state: int, next_state: int, time_step: int) -> float:
        """
        Calculates bonus for the current transition.
        
        Args:
            prev_state: Previous state
            next_state: State we're transitioning to
            time_step: Current time step
            
        Returns:
            bonus score
        """
        current_transition = (prev_state, next_state)
        
        if current_transition in required_transitions:
            return bonus_per_transition
        
        return 0.0
    
    return bonus_fn

def create_sequence_bonus_incentive_vectorized(
    constraint_length: int,
    bonus_per_transition: float = 2.0,
    device: torch.device = torch.device('cpu'),
    num_states: int = None  # Add this parameter
) -> Tuple[torch.Tensor, Callable]:
    """
    Creates a precomputed bonus matrix for faster lookup.
    
    Args:
        constraint_length: Length of consecutive sequence to encourage
        bonus_per_transition: Bonus score for each required transition
        device: Device for tensor
        num_states: Total number of states in HMM (if None, uses constraint_length)
    
    Returns:
        bonus_matrix: (N, N) tensor where bonus_matrix[i,j] contains the bonus
        bonus_fn: Original function signature for compatibility
    """
    # If num_states not specified, use constraint_length as minimum
    if num_states is None:
        num_states = constraint_length
    else:
        num_states = max(num_states, constraint_length)
    
    # Create a matrix of bonuses
    bonus_matrix = torch.zeros((num_states, num_states), device=device)
    
    # Set bonuses for required transitions (i, i+1)
    for i in range(constraint_length - 1):
        bonus_matrix[i, i+1] = bonus_per_transition
    
    def bonus_fn(prev_state: int, next_state: int, time_step: int) -> float:
        if prev_state < bonus_matrix.shape[0] and next_state < bonus_matrix.shape[1]:
            return bonus_matrix[prev_state, next_state].item()
        return 0.0
    
    return bonus_matrix, bonus_fn