import numpy as np
from typing import Optional, Tuple, List


######################################################################
# HMM Generation and Helper Functions
######################################################################


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def generate_random_hmm(
    num_states: int,
    num_emissions: int,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generates random HMM parameters as probabilities (not log probabilities).

    Args:
        num_states: Number of hidden states (N)
        num_emissions: Number of possible emission symbols (M)
        seed: Optional random seed for reproducibility

    Returns:
        List of [transition_matrix, emission_matrix, initial_vector] as probabilities
    """
    rng = np.random.default_rng(seed)

    # Generate random transition matrix (N x N)
    transition_logits = rng.standard_normal((num_states, num_states))
    transition_probs = _softmax(transition_logits, axis=1)

    # Generate random emission matrix (N x M)
    emission_logits = rng.standard_normal((num_states, num_emissions))
    emission_probs = _softmax(emission_logits, axis=1)

    # Generate random initial state distribution (N,)
    initial_logits = rng.standard_normal(num_states)
    initial_probs = _softmax(initial_logits, axis=0)

    return [transition_probs, emission_probs, initial_probs]


def hmm_log_probability(hidden_sequence, emission_sequence, initial, transition, emission):
    """
    Compute the log probability of a hidden state and emission sequence given HMM parameters.

    Args:
        hidden_sequence: 1D array-like of ints representing hidden state sequence
        emission_sequence: 1D array-like of ints representing observed emission sequence
        initial: array of shape (num_states,) - initial state probabilities
        transition: array of shape (num_states, num_states) - transition probabilities
                   transition[i, j] = P(state_j | state_i)
        emission: array of shape (num_states, num_observations) - emission probabilities
                 emission[i, k] = P(observation_k | state_i)

    Returns:
        log_prob: scalar float containing log probability of the sequences
    """
    hidden_sequence = np.asarray(hidden_sequence, dtype=int)
    emission_sequence = np.asarray(emission_sequence, dtype=int)
    initial = np.asarray(initial, dtype=float)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)

    log_initial = np.log(initial)
    log_transition = np.log(transition)
    log_emission = np.log(emission)

    seq_length = len(hidden_sequence)

    log_prob = log_initial[hidden_sequence[0]]
    log_prob += log_emission[hidden_sequence[0], emission_sequence[0]]

    for t in range(1, seq_length):
        log_prob += log_transition[hidden_sequence[t - 1], hidden_sequence[t]]
        log_prob += log_emission[hidden_sequence[t], emission_sequence[t]]

    return float(log_prob)


def sample_from_hmm(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_vector: np.ndarray,
    sequence_length: int,
    constraint_length: Optional[int] = None,
    seed: Optional[int] = None,
    max_attempts: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a single sample sequence from an HMM.
    If constraint_length is specified, injects the sequence [0,1,2,...,N-1] when state 0 is encountered.

    Args:
        transition_matrix: Transition probabilities (N x N)
        emission_matrix: Emission probabilities (N x M)
        initial_vector: Initial state probabilities (N,)
        sequence_length: Length of sequence to generate (T)
        constraint_length: If specified, injects sequence [0,1,...,N-1] when 0 is sampled
        seed: Optional random seed for reproducibility
        max_attempts: Maximum number of sampling attempts before giving up

    Returns:
        states: Hidden state sequence (T,)
        observations: Observation sequence (T,)
    """
    rng = np.random.default_rng(seed)

    initial_probs = np.asarray(initial_vector, dtype=float)
    transition_probs = np.asarray(transition_matrix, dtype=float)
    emission_probs = np.asarray(emission_matrix, dtype=float)

    for attempt in range(max_attempts):
        states = []
        observations = []
        constraint_injected = False

        # Sample initial state
        current_state = rng.choice(len(initial_probs), p=initial_probs)
        states.append(current_state)

        # Sample initial observation
        obs = rng.choice(emission_probs.shape[1], p=emission_probs[current_state])
        observations.append(obs)

        # Check if we should inject constraint at the start
        if constraint_length is not None and current_state == 0 and not constraint_injected:
            for i in range(1, constraint_length):
                states.append(i)
                obs = rng.choice(emission_probs.shape[1], p=emission_probs[i])
                observations.append(obs)

            current_state = constraint_length - 1
            constraint_injected = True

            if len(states) > sequence_length:
                continue

        # Sample remaining sequence
        while len(states) < sequence_length:
            current_state = rng.choice(transition_probs.shape[1], p=transition_probs[current_state])
            states.append(current_state)

            obs = rng.choice(emission_probs.shape[1], p=emission_probs[current_state])
            observations.append(obs)

            if constraint_length is not None and current_state == 0 and not constraint_injected:
                for i in range(1, constraint_length):
                    states.append(i)
                    obs = rng.choice(emission_probs.shape[1], p=emission_probs[i])
                    observations.append(obs)

                current_state = constraint_length - 1
                constraint_injected = True

                if len(states) > sequence_length:
                    break

        if len(states) == sequence_length:
            if constraint_length is None or constraint_injected:
                return np.array(states, dtype=int), np.array(observations, dtype=int)

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
    """
    states = np.asarray(states, dtype=int)
    max_length = 0

    for i in range(len(states)):
        if states[i] == start_state:
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
    states: np.ndarray,
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
    states = np.asarray(states, dtype=int)

    max_length = 0
    best_endpoint = None

    for i in range(len(states)):
        if states[i] == 0:
            length = 1
            j = i + 1
            expected_state = 1

            while j < len(states) and states[j] == expected_state:
                length += 1
                expected_state += 1
                j += 1

            if length > max_length:
                max_length = length
                best_endpoint = i + length - 1

    return best_endpoint