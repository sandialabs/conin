import numpy as np
import random
import pprint
import math
import json
import ast

from conin import InvalidInputError
from conin.hmm import Internal_HMM
from conin.hmm import Statistical_Model
from conin.util import Util


class HMM(Statistical_Model):
    def __init__(self):
        """
        Initializes the Hidden Markov Model (HMM) with empty parameters.

        Attributes:
            start_vec (list): Vector of starting probabilities for hidden states.
            transition_mat (list): Matrix of transition probabilities between hidden states.
            emission_mat (list): Matrix of emission probabilities from hidden states to observed states.
            hidden_to_internal (dict): Maps hidden state labels to indices.
            observed_to_internal (dict): Maps observed state labels to indices.
            hidden_to_external (list): Inverse mapping from indices to hidden state labels.
            observed_to_external (list): Inverse mapping from indices to observed state labels.
            num_hidden_states (int): Number of hidden states.
            num_observed_states (int): Number of observed states.
        """
        self.internal_hmm = None
        self.start_vec = []
        self.transition_mat = []
        self.emission_mat = []
        self.hidden_to_internal = {}  # Maps from labels to indices
        self.observed_to_internal = {}
        self.hidden_to_external = []  # Maps from labels to indices
        self.observed_to_external = []
        self.num_hidden_states = None  # Number of hidden variables
        self.num_observed_states = None  # Number of observed variables

    def __str__(self):
        """
        Nice printing
        """
        return pprint.pformat(self.to_dict(), indent=4, sort_dicts=True)

    def get_hmm(self):
        """
        Returns hmm
        """
        return self

    def get_internal_hmm(self):
        """
        Returns internal hmm
        """
        return self.internal_hmm

    def load_model(self, *, start_probs, transition_probs, emission_probs):
        """
        Loads the model parameters including starting probabilities, transition probabilities, and emission probabilities.

        Parameters:
            start_probs (dict): A dictionary mapping hidden states to their starting probabilities.
            transition_probs (dict): A dictionary mapping pairs of hidden states to their transition probabilities.
            emission_probs (dict): A dictionary mapping pairs of hidden states and observed states to their emission probabilities.

        Raises:
            InvalidInputError: If any probabilities are negative or if the probabilities do not sum to 1.
        """
        self.internal_hmm = None
        self.start_vec = []
        self.transition_mat = []
        self.emission_mat = []
        self.hidden_to_internal = {}  # Maps from labels to indices
        self.observed_to_internal = {}
        self.hidden_to_external = []  # Maps from labels to indices
        self.observed_to_external = []
        self.num_hidden_states = None  # Number of hidden variables
        self.num_observed_states = None  # Number of observed variables

        # Setup hidden_to_internal, hidden_to_external, and num_hidden_states
        # NOTE: Sorting added here to simplify debugging
        for h1, h2 in sorted(transition_probs.keys()):
            if h1 not in self.hidden_to_internal.keys():
                self.hidden_to_external.append(h1)
                self.hidden_to_internal[h1] = len(self.hidden_to_external) - 1
            if h2 not in self.hidden_to_internal.keys():
                self.hidden_to_external.append(h2)
                self.hidden_to_internal[h2] = len(self.hidden_to_external) - 1
        self.num_hidden_states = len(self.hidden_to_internal)

        # Setup observed_to_internal, observed_to_external, and num_observed_states
        for h, o in emission_probs:
            if o not in self.observed_to_internal:
                self.observed_to_external.append(o)
                self.observed_to_internal[o] = len(self.observed_to_external) - 1
        self.num_observed_states = len(self.observed_to_internal)

        # Setup start_vec
        self.start_vec = [0] * self.num_hidden_states
        for h, prob in start_probs.items():
            self.start_vec[self.hidden_to_internal[h]] = prob

        # Setup transition_mat
        self.transition_mat = [
            [0 for _ in range(self.num_hidden_states)]
            for _ in range(self.num_hidden_states)
        ]
        for (h1, h2), prob in transition_probs.items():
            # No new hidden states
            if (h1 not in self.hidden_to_internal) or (
                h2 not in self.hidden_to_internal
            ):
                raise InvalidInputError("You shouldn't see this")
            self.transition_mat[self.hidden_to_internal[h1]][
                self.hidden_to_internal[h2]
            ] = prob

        # Setup emission_mat
        self.emission_mat = [
            [0 for _ in range(self.num_observed_states)]
            for _ in range(self.num_hidden_states)
        ]
        for (h, o), prob in emission_probs.items():
            # No new hidden states
            if h not in self.hidden_to_internal:
                raise InvalidInputError(
                    "start_probs does not contain all hidden states appearing in emission_probs, (",
                    h,
                    ", ",
                    o,
                    "): ",
                    prob,
                )
            self.emission_mat[self.hidden_to_internal[h]][
                self.observed_to_internal[o]
            ] = prob

        self.internal_hmm = Internal_HMM(
            start_vec=self.start_vec,
            transition_mat=self.transition_mat,
            emission_mat=self.emission_mat,
        )

    def is_valid_observed_state(self, o):
        """
        Check if the given observed state is allowed

        Parameters:
            o : Any: The observed state to be checked for validity.

        Returns:
            bool: True if the hidden state `h` is valid (i.e., exists in the mapping),
            False otherwise.
        """
        return o in self.observed_to_internal

    def is_valid_hidden_state(self, h):
        """
        Check if the given hidden state is allowed

        Parameters:
            h : Any: The hidden state to be checked for validity.

        Returns:
            bool: True if the hidden state `h` is valid (i.e., exists in the mapping),
            False otherwise.
        """
        return h in self.hidden_to_internal

    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Parameters:
            seed (int): The seed value to set for the random number generator.
        """
        np.random.seed(seed)

    def get_hidden_states(self):
        """
        Returns:
            set: hidden states
        """
        return set(self.hidden_to_external)

    def get_observable_states(self):
        """
        Returns:
            set: observable states
        """
        return set(self.observed_to_external)

    def get_start_probs(self):
        """
        Retrieves the starting probabilities of the hidden states.

        Returns:
            dict: A dictionary mapping hidden states to their starting probabilities.
        """
        # Same format as in load_model
        return {
            self.hidden_to_external[h]: self.start_vec[h]
            for h in range(self.num_hidden_states)
        }

    def get_transition_probs(self):
        # Same format as in load_model
        """
        Retrieves the transition probabilities between hidden states.

        Returns:
            dict: A dictionary mapping pairs of hidden states to their transition probabilities.
        """
        return {
            (
                self.hidden_to_external[h1],
                self.hidden_to_external[h2],
            ): self.transition_mat[h1][h2]
            for h1 in range(self.num_hidden_states)
            for h2 in range(self.num_hidden_states)
        }

    def get_emission_probs(self):
        """
        Retrieves the emission probabilities from hidden states to observed states.

        Returns:
            dict: A dictionary mapping pairs of hidden states and observed states to their emission probabilities.
        """
        # Same format as in load_model
        return {
            (
                self.hidden_to_external[h],
                self.observed_to_external[o],
            ): self.emission_mat[h][o]
            for h in range(self.num_hidden_states)
            for o in range(self.num_observed_states)
        }

    def to_dict(self):
        """
        Generate a dict representation of the model data.

        Returns:
            dict: A dictionary representaiton of this statistical model.
        """

        start_probs = {
            self.hidden_to_external[i]: v
            for i, v in enumerate(self.start_vec)
            if v > 1e-3
        }
        transition_probs = {
            (self.hidden_to_external[i], self.hidden_to_external[j]): v
            for i, row in enumerate(self.transition_mat)
            for j, v in enumerate(row)
            if v > 1e-3
        }
        emission_probs = {
            (self.hidden_to_external[i], self.observed_to_external[o]): v
            for i, row in enumerate(self.emission_mat)
            for o, v in enumerate(row)
            if v > 1e-3
        }

        return dict(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            num_hidden=self.num_hidden_states,
            num_observed=self.num_observed_states,
        )

    # Generation
    def generate_hidden(self, time_steps):
        """
        Generates a sequence of hidden variables satisfying the internal constraints.

        Parameters:
            time_steps (int): How long you want the sequence to be.

        Returns:
            list: Feasible sequence of hidden states (labels).
        """
        internal_hidden = self.internal_hmm.generate_hidden(time_steps)
        return [self.hidden_to_external[h] for h in internal_hidden]

    def generate_hidden_until_state(self, h):
        """
        Generates a sequence of hidden variables which stops at the first time step where we reach hidden state h.

        Parameters:
            h: stopping hidden state:

        Returns:
            list: Feasible sequence of hidden states, where last value is h
        """
        internal_hidden = self.internal_hmm.generate_hidden_until_state(
            self.hidden_to_internal[h]
        )
        return [self.hidden_to_external[h] for h in internal_hidden]

    def generate_observed_from_hidden(self, hidden):
        """
        Generates a random observed sequence of states from a hidden sequence of states.

        Parameters:
            hidden (list): A list of hidden states from which to generate observations.

        Returns:
            list: Observations generated from the hidden states.
        """
        internal_hidden = [self.hidden_to_internal[h] for h in hidden]
        internal_observed = self.internal_hmm.generate_observed_from_hidden(
            internal_hidden
        )
        return [self.observed_to_external[o] for o in internal_observed]

    def generate_observed(self, time_steps):
        """
        Generates a random observed sequence of states.

        Parameters:
            time_steps (int): The number of time steps for which to generate observed states.

        Returns:
            list: Observations generated from the hidden states.
        """
        internal_observed = self.internal_hmm.generate_observed(time_steps)
        return [self.observed_to_external[o] for o in internal_observed]

    def log_probability(self, observations, hidden):
        """
        Compute the log-probability of the observations given the hidden state.
        """

        h = [self.hidden_to_internal[hval] for hval in hidden]
        o = [self.observed_to_internal[oval] for oval in observations]

        ans = math.log(self.start_vec[h[0]]) + math.log(self.emission_mat[h[0]][o[0]])
        for t in range(1, len(observations)):
            ans += math.log(self.transition_mat[h[t - 1]][h[t]]) + math.log(
                self.emission_mat[h[t]][o[t]]
            )

        return ans

    def make_non_zero(self, tol=1e-6):
        """
        Takes all nonzero parameters and sets them to tolerance, then renormalizing everything

        Parameters:
            tol: what we set the zero values to
        """
        start_probs = self.get_start_probs()
        transition_probs = self.get_transition_probs()
        emission_probs = self.get_emission_probs()

        for h, val in start_probs.items():
            if val < tol:
                start_probs[h] = tol
        for key, val in transition_probs.items():
            if val < tol:
                transition_probs[key] = tol
        for key, val in emission_probs.items():
            if val < tol:
                emission_probs[key] = tol

        start_probs = Util.normalize_dictionary(start_probs)
        transition_probs = Util.normalize_2d_dictionary(transition_probs)
        emission_probs = Util.normalize_2d_dictionary(emission_probs)

        self.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )

    def write_to_file(self, file_name):
        """
        Writes the hmm to a file

        Parameters:
            file_name: Name of file we are writing to
        """
        start_probs = self.get_start_probs()
        transition_probs = self.get_transition_probs()
        emission_probs = self.get_emission_probs()

        # Convert tuples to strings for JSON serialization
        transition_probs_serializable = {str(k): v for k, v in transition_probs.items()}
        emission_probs_serializable = {str(k): v for k, v in emission_probs.items()}

        # Create a dictionary to hold all the data
        data = {
            "start_probs": start_probs,
            "transition_probs": transition_probs_serializable,
            "emission_probs": emission_probs_serializable,
        }

        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def read_from_file(self, file_name):
        """
        Reads the hmm from a file and returns the dictionaries.

        Parameters:
            file_name: Name of the file we are reading from
        """

        # Read the data from the JSON file
        with open(file_name, "r") as json_file:
            data = json.load(json_file)

        # Convert string keys back to tuples
        transition_probs = {
            ast.literal_eval(k): v for k, v in data["transition_probs"].items()
        }
        emission_probs = {
            ast.literal_eval(k): v for k, v in data["emission_probs"].items()
        }

        # Extract start probabilities
        start_probs = data["start_probs"]

        self.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )
