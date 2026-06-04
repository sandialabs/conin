from munch import Munch
import copy
import json

from algorithms import hmm_constrained_inference
# from ICML_ILP import hmm_constrained_inference
from ICML_comparisons import generate_random_hmm, sample_from_hmm, find_longest_consecutive_sequence


#############################
# Experiment Parameters
#############################


batch = 25 #25 runs per batch
num_states = 100 #hidden states
num_emissions = 10 #emission states
constraint_length = 10 #length of required subsequence
seq_lengths = [100*i for i in range(1,11)] #time horizon

#Generate HMM
hmm_params = generate_random_hmm(num_states, num_emissions)
transition_matrix, emission_matrix, initial_vector = hmm_params


#############################
# Run the Experiment
#############################

length_exp_ilp = []
fail_list = []

for length in seq_lengths:
    #Initialize list of times
    times_ilp = []
    #Runtime experiments
    fail_ctr = 0
    for b in range(batch):
        if b % 10 == 0:
            print(f'On length {length} and run {b}')
 
        states, observations = sample_from_hmm(
            transition_matrix, emission_matrix, initial_vector,
            sequence_length=length,
            constraint_length=constraint_length  # Inject consecutive sequence when 0 is encountered
        )

                
        #ILP
        states, solve_time, success = hmm_constrained_inference(
            observations, transition_matrix, emission_matrix, initial_vector, constraint_length
        )
        times_ilp.append(solve_time)


        # Check that solution is feasible
        failed_length = False
        if success:
            max_length = find_longest_consecutive_sequence(states)
            failed_length = max_length < constraint_length
        if not success or failed_length: #increment by 1 if failure at any point
            fail_ctr += 1
            print(f'Infeasible solution on experiment setting {length} and batch {b}')

    length_exp_ilp.append(times_ilp)
    fail_list.append(fail_ctr)

data = {'seq_lengths': seq_lengths, 
        'batch_size': batch,
        'constraint_length': constraint_length,
       'hmm_details': [num_states, num_emissions ],
       'runtimes': length_exp_ilp,
       'num_fails': fail_list}

with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

print('Finished Running!')