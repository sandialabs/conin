import numpy as np
import torch
import json
from munch import Munch
import itertools
from collections import defaultdict
import random
import copy
import pickle
import matplotlib.pyplot as plt

import importlib
import time

import pulp
import time

from typing import Callable, Optional, Tuple, List, Tuple

######################################################################
# HMM Generation and Helper Functions
######################################################################


def generate_random_hmm(
    num_states: int,
    num_emissions: int,
    seed: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Generates random HMM parameters as probabilities (not log probabilities).
    
    Args:
        num_states: Number of hidden states (N)
        num_emissions: Number of possible emission symbols (M)
        seed: Optional random seed for reproducibility
        
    Returns:
        List of [transition_matrix, emission_matrix, initial_vector] as probabilities
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random transition matrix (N x N)
    # Sample from Dirichlet-like distribution using softmax of random values
    transition_logits = torch.randn(num_states, num_states)
    transition_probs = torch.softmax(transition_logits, dim=1)
    
    # Generate random emission matrix (N x M)
    emission_logits = torch.randn(num_states, num_emissions)
    emission_probs = torch.softmax(emission_logits, dim=1)
    
    # Generate random initial state distribution (N,)
    initial_logits = torch.randn(num_states)
    initial_probs = torch.softmax(initial_logits, dim=0)
    
    return [transition_probs, emission_probs, initial_probs]

def hmm_log_probability(hidden_sequence, emission_sequence, initial, transition, emission):
    """
    Compute the log probability of a hidden state and emission sequence given HMM parameters.
    
    Args:
        hidden_sequence: 1D tensor or list of ints representing hidden state sequence
        emission_sequence: 1D tensor or list of ints representing observed emission sequence
        initial: tensor of shape (num_states,) - initial state probabilities
        transition: tensor of shape (num_states, num_states) - transition probabilities
                   transition[i, j] = P(state_j | state_i)
        emission: tensor of shape (num_states, num_observations) - emission probabilities
                 emission[i, k] = P(observation_k | state_i)
    
    Returns:
        log_prob: scalar tensor containing log probability of the sequences
    """
    # Convert sequences to tensors if they're lists
    if isinstance(hidden_sequence, list):
        hidden_sequence = torch.tensor(hidden_sequence, dtype=torch.long)
    if isinstance(emission_sequence, list):
        emission_sequence = torch.tensor(emission_sequence, dtype=torch.long)
    
    # Convert probabilities to log space for numerical stability
    log_initial = torch.log(initial)
    log_transition = torch.log(transition)
    log_emission = torch.log(emission)
    
    seq_length = len(hidden_sequence)
    
    # Log probability = log P(initial state) + sum of log transitions + sum of log emissions
    
    # 1. Initial state probability
    log_prob = log_initial[hidden_sequence[0]]
    
    # 2. Emission probability for first state
    log_prob += log_emission[hidden_sequence[0], emission_sequence[0]]
    
    # 3. Transition and emission probabilities for remaining states
    for t in range(1, seq_length):
        # Transition from state at t-1 to state at t
        log_prob += log_transition[hidden_sequence[t-1], hidden_sequence[t]]
        # Emission at state t
        log_prob += log_emission[hidden_sequence[t], emission_sequence[t]]
    
    return log_prob



def sample_from_hmm(
    transition_matrix: torch.Tensor,
    emission_matrix: torch.Tensor,
    initial_vector: torch.Tensor,
    sequence_length: int,
    constraint_length: Optional[int] = None,
    seed: Optional[int] = None,
    max_attempts: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw a single sample sequence from an HMM.
    If constraint_length is specified, injects the sequence [0,1,2,...,N-1] when state 0 is encountered.
    
    Args:
        transition_matrix: Transition probabilities (N x N) - NOT log probabilities
        emission_matrix: Emission probabilities (N x M) - NOT log probabilities
        initial_vector: Initial state probabilities (N,) - NOT log probabilities
        sequence_length: Length of sequence to generate (T)
        constraint_length: If specified, injects sequence [0,1,...,N-1] when 0 is sampled
        seed: Optional random seed for reproducibility
        max_attempts: Maximum number of sampling attempts before giving up
        
    Returns:
        states: Hidden state sequence (T,)
        observations: Observation sequence (T,)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Use probabilities directly (no need to convert from log space)
    initial_probs = initial_vector
    transition_probs = transition_matrix
    emission_probs = emission_matrix
    
    for attempt in range(max_attempts):
        states = []
        observations = []
        constraint_injected = False
        
        # Sample initial state
        current_state = torch.multinomial(initial_probs, 1).item()
        states.append(current_state)
        
        # Sample initial observation
        obs = torch.multinomial(emission_probs[current_state], 1).item()
        observations.append(obs)
        
        # Check if we should inject constraint at the start
        if constraint_length is not None and current_state == 0 and not constraint_injected:
            # Inject the full sequence [0, 1, 2, ..., constraint_length-1]
            for i in range(1, constraint_length):
                states.append(i)
                obs = torch.multinomial(emission_probs[i], 1).item()
                observations.append(obs)
            
            current_state = constraint_length - 1
            constraint_injected = True
            
            # Check if injection made sequence too long
            if len(states) > sequence_length:
                continue  # Resample
        
        # Sample remaining sequence
        while len(states) < sequence_length:
            # Sample next state given current state
            current_state = torch.multinomial(transition_probs[current_state], 1).item()
            states.append(current_state)
            
            # Sample observation given current state
            obs = torch.multinomial(emission_probs[current_state], 1).item()
            observations.append(obs)
            
            # Check if we encountered 0 and should inject constraint
            if constraint_length is not None and current_state == 0 and not constraint_injected:
                # Inject the full sequence [0, 1, 2, ..., constraint_length-1]
                for i in range(1, constraint_length):
                    states.append(i)
                    obs = torch.multinomial(emission_probs[i], 1).item()
                    observations.append(obs)
                
                current_state = constraint_length - 1
                constraint_injected = True
                
                # Check if injection made sequence too long
                if len(states) > sequence_length:
                    break  # Break inner loop to resample
        
        # Check if sequence is valid length
        if len(states) == sequence_length:
            # If constraint was specified, verify it was injected
            if constraint_length is None or constraint_injected:
                return torch.tensor(states), torch.tensor(observations)
    
    # Failed to find a valid sequence
    raise RuntimeError(
        f"Failed to sample a valid sequence after {max_attempts} attempts. "
        f"Try increasing sequence_length or max_attempts."
    )


def find_longest_consecutive_sequence(
    states: List[int],
    start_state: int = 0,
    constraint_length: int = None
) -> int:
    """
    Find the maximum length of a contiguous consecutive sequence [start, start+1, ..., m]
    occurring anywhere in the path.
    
    Args:
        states: State sequence (list or tensor)
        start_state: Starting state of the sequence to look for (default: 0)
        
    Returns:
        Maximum length of consecutive sequence found (e.g., 5 if [0,1,2,3,4] is found)
    """
    max_length = 0
    
    for i in range(len(states)):
        # Check if we're at the start state
        if states[i] == start_state:
            # Count how long the consecutive sequence continues
            length = 1
            j = i + 1
            expected_state = start_state + 1
            
            while j < len(states) and states[j] == expected_state:
                length += 1
                expected_state += 1
                j += 1
            
            max_length = max(max_length, length)
    
    
    if constraint_length:
        max_length = min(max_length, constraint_length)
    return max_length


def find_constraint_endpoint(
    states: torch.Tensor,
    constraint_length: int
) -> Optional[int]:
    """
    Find the index where the longest consecutive sequence starting from 0 ends.
    Looks for sequences [0], [0,1], [0,1,2], etc. and returns the endpoint of the longest one found.
    
    Args:
        states: Hidden state sequence (T,)
        constraint_length: Target length of the consecutive sequence (N)
        
    Returns:
        Index where the longest consecutive sequence ends, or None if state 0 is never found
    """
    max_length = 0
    best_endpoint = None
    
    # Scan through the sequence looking for state 0
    for i in range(len(states)):
        if states[i].item() == 0:
            # Count how long the consecutive sequence continues from here
            length = 1
            j = i + 1
            expected_state = 1
            
            while j < len(states) and states[j].item() == expected_state:
                length += 1
                expected_state += 1
                j += 1
            
            # Update if this is the longest sequence found
            if length > max_length:
                max_length = length
                best_endpoint = i + length - 1
    
    return best_endpoint







