import torch
import ICML_ILP
import ICML_conin

# Example usage - can be run directly in Jupyter
# Simple example: 5 states, 2 observation symbols, sequence length 15
K = 10  # number of states
M = 5  # number of observation symbols
T = 15  # sequence length
N = 4  # require subsequence [0, 1, 2, 3] to occur

# Create random HMM parameters (in log space)
torch.manual_seed(31)
transition = torch.randn(K, K)
emission = torch.randn(K, M)
initial = torch.randn(K)

# Random observations
observations = torch.randint(0, M, (T,))

print("Observations:", observations.tolist())
print(f"Required subsequence: [0, 1, ..., {N-1}]")


# Run constrained inference
states, solve_time = ICML_ILP.hmm_constrained_inference(
    observations, transition, emission, initial, N
)

print("Inferred states:", states.tolist())
print(f"ILP solve time: {solve_time:.4f} seconds")


# Run constrained inference
states, solve_time = ICML_conin.hmm_constrained_inference(
    observations, transition, emission, initial, N
)

print("Inferred states:", states.tolist())
print(f"ILP solve time: {solve_time:.4f} seconds")

