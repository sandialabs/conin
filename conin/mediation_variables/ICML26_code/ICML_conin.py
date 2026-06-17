import math
from typing import Callable, Optional, Tuple, List, Tuple
import time
import numpy as np
import torch

import pyomo.environ as pyo
import conin
import conin.inference
from conin.hidden_markov_model import HiddenMarkovModel, ConstrainedHiddenMarkovModel


######################################################################
# ILP
######################################################################


def hmm_constrained_inference(
    observations: torch.Tensor,
    transition: torch.Tensor,
    emission: torch.Tensor,
    initial: torch.Tensor,
    N: int,
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
    M = emission.shape[1]  # number of emissions

    # Convert to numpy and ensure we're working with log probabilities
    trans = transition.detach().cpu().numpy()
    emit = emission.detach().cpu().numpy()
    init = initial.detach().cpu().numpy()
    obs = observations.detach().cpu().numpy()

    start_probs = {k:math.exp(init[k]) for k in range(K)}
    emission_probs = {(k,m):math.exp(emit[k,m]) for k in range(K) for m in range(M)}
    transition_probs = {(k1,k2):math.exp(trans[k1,k2]) for k1 in range(K) for k2 in range(K)}

    #import pprint
    #pprint.pprint(start_probs)
    #pprint.pprint(emission_probs)
    #pprint.pprint(transition_probs)

    hmm = HiddenMarkovModel()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )

    #
    # Setup constrained HMM
    #
    # Add subsequence constraint: [0, 1, 2, ..., N-1] must occur somewhere
    #
    if N > 0:
        # Check if sequence is feasible
        if N > T:
            raise ValueError(
                f"Subsequence length N={N} cannot be longer than sequence length T={T}"
            )
        if N > K:
            raise ValueError(
                f"Subsequence length N={N} cannot be longer than number of states K={K}"
            )

        @conin.pyomo_constraint_fn()
        def subsequence_constraints(M, D):
            # Binary variables indicating if the subsequence starts at position t
            M.start = pyo.Var(list(range(T - N + 1)), domain=pyo.Binary)

            # The subsequence starts at least once
            M.start_once = pyo.Constraint(expr= sum(M.start[t] for t in range(T-N+1)) >= 1)

            M.c = pyo.ConstraintList()
            for t in range(T - N + 1):  # Valid starting positions
                # subseq_starts[t] = 1 iff the full subsequence [0,1,...,N-1] occurs starting at t
                # This means: z[t, 0] = 1 AND z[t+1, 1] = 1 AND ... AND z[t+N-1, N-1] = 1
                for i in range(N):
                    M.c.add(M.start[t] <= M.V("H", t + i, i))

                # If all positions match, subseq_starts[t] can be 1
                #M.c.add( M.start[t] >= sum(M.V("H", t+i, i) for i in range(N)) - N + 1 )

            M.pprint()

        chmm = ConstrainedHiddenMarkovModel(
            hmm=hmm, constraints=[subsequence_constraints]
        )
        chmm.initialize_chmm()
        pgm = chmm
    else:
        pgm = hmm

    # Create an inference wrapper for a dynamic probabilistic graphical model, using integer programming
    start_time = time.time()
    inference = conin.inference.DPGM_IntegerProgrammingInference(pgm)
    results = inference.map_query(evidence=list(obs), solver="gurobi", ip_formulation="network_flow")
    solution = results.solution.states
    solve_time = time.time() - start_time

    return torch.tensor(solution, dtype=torch.long), solve_time
