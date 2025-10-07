import numpy as np
import random
import pprint
import math
import json
import ast

from conin.exceptions import InvalidInputError
#from conin.hmm import Statistical_Model
#from conin.hmm import Internal_Statistical_Model
from conin.util import Util


class HMM:

    def __init__(self, *, start_vec, transition_mat, emission_mat, check_errors=True):
        self.load_start_vec(start_vec, check_errors=check_errors)
        self.load_transition_mat(transition_mat, check_errors=check_errors)
        self.load_emission_mat(emission_mat, check_errors=check_errors)
        if check_errors:
            self.check_dimensions()
        self.load_dimensions()

    def Xget_internal_hmm(self):
        return self

    def load_start_vec(self, start_vec, check_errors=True):
        """
        Loads the start vec

        Parameters:
            start_vec (array): Starting Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the probabilities do not sum to 1
        """
        if check_errors:
            # Confirm that the start_vec is non-negative
            for prob in start_vec:
                if not prob >= 0:
                    raise InvalidInputError(
                        "start_probs values must be positive floats."
                    )
            # Confirm that the start_vec sums to one
            if not np.isclose(sum(start_vec), 1):
                raise InvalidInputError("start_prob values must sum to 1.")

        self.start_vec = start_vec

    def load_transition_mat(self, transition_mat, check_errors=True):
        """
        Loads the transition matrix

        Parameters:
            transition_mat (array): Transition Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the rows do not sum to 1
        """
        if check_errors:
            # Non-negative transition probabilities
            for h1 in range(len(transition_mat)):
                for h2 in range(len(transition_mat[h1])):
                    if not transition_mat[h1][h2] >= 0:
                        raise InvalidInputError(
                            "Transition_mat must be positive floats."
                        )
            # Rows sum to 1
            for vec in transition_mat:
                if not np.isclose(sum(vec), 1):
                    raise InvalidInputError("Transition_mat rows do not sum to 1.")

        self.transition_mat = transition_mat

    def load_emission_mat(self, emission_mat, check_errors=True):
        """
        Loads the emission matrix

        Parameters:
            emission_mat (array): Emission Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the rows do not sum to 1
        """
        if check_errors:
            # Non-negative emission probabilities
            for h1 in range(len(emission_mat)):
                for h2 in range(len(emission_mat[h1])):
                    if not emission_mat[h1][h2] >= 0:
                        raise InvalidInputError("Emission_mat must be positive floats.")
            # Rows sum to 1
            for vec in emission_mat:
                if not np.isclose(sum(vec), 1):
                    raise InvalidInputError(
                        f"Emission_mat rows do not sum to 1: {sum(vec)}"
                    )

        self.emission_mat = emission_mat

    def check_dimensions(self):
        """
        TODO: do we actually need this if internal_hmm is only called by hmm?

        Checks that the dimensions of the input matrices are appropriately sized

        Raises:
            InvalidInputError: If any dimensions do not line up correctly
        """
        correct_dimension = True

        if len(self.start_vec) != len(self.transition_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self.transition_mat:
            if len(self.start_vec) != len(vec):
                correct_dimension = False  # pragma: no cover

        if len(self.start_vec) != len(self.emission_mat):
            correct_dimension = False  # pragma: no cover
        for vec in self.emission_mat:
            if len(self.emission_mat[0]) != len(vec):
                correct_dimension = False  # pragma: no cover

        if not correct_dimension:
            raise InvalidInputError(
                "Dimensions do not line up correctly in check_dimensions, you shouldn't see this."  # pragma: no cover
            )

    def load_dimensions(self):
        """
        Updates num_hidden and num_observed
        """
        self.num_hidden_states = len(self.start_vec)
        self.num_observed_states = len(self.emission_mat[0])
        self.hidden_states = range(self.num_hidden_states)
        self.observed_states = range(self.num_observed_states)

    def generate_hidden(self, time_steps):
        """
        Generates a sequence of hidden states based on the model's parameters.

        Parameters:
            time_steps (int): The number of time steps for which to generate hidden states.

        Returns:
            list: A list of indices representing the generated hidden states.

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_hidden time_steps > 0.")
        hidden = []

        # Sample the first hidden state
        hidden.append(Util.sample_from_vec(self.start_vec))

        # Sample subsequent hidden states
        for t in range(time_steps - 1):
            hidden.append(Util.sample_from_vec(self.transition_mat[hidden[t]]))

        return hidden

    def generate_hidden_until_state(self, h):
        """
        Generates a sequence of hidden variables which stops at the first time step where we reach hidden state h.

        Parameters:
            h: stopping hidden state:

        Returns:
            list: Feasible sequence of hidden states, where last value is h
        """
        hidden = []

        # Sample the first hidden state
        hidden.append(Util.sample_from_vec(self.start_vec))

        # Sample until the last hidden state is h
        while hidden[-1] != h:
            hidden.append(Util.sample_from_vec(self.transition_mat[hidden[-1]]))

        return hidden

    def generate_observed_from_hidden(self, hidden):
        """
        Generates a sequence of observed states from a given sequence of hidden states.

        Parameters:
            hidden (list): A list of indices representing hidden states.

        Returns:
            list: A list of indices representing the generated observed states.

        """
        observed = []
        time_steps = len(hidden)

        for t in range(time_steps):
            observed.append(Util.sample_from_vec(self.emission_mat[hidden[t]]))

        return observed

    def generate_observed(self, time_steps):
        """
        Generates a sequence of observed states based on the model's parameters.

        Parameters:
            time_steps (int): The number of time steps for which to generate observed states.

        Returns:
            list: A list of indices representing the generated observed states.

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_observed time_steps > 0.")
        hidden = self.generate_hidden(time_steps)
        return self.generate_observed_from_hidden(hidden)


class HiddenMarkovModel:

    def __init__(self):
        """
        Initializes the Hidden Markov Model with empty parameters.

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
        self._hmm = None
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

    def Xget_hmm(self):
        """
        Returns hmm
        """
        return self

    def Xget_internal_hmm(self):
        """
        Returns internal hmm
        """
        return self._hmm

    @property
    def internal_hmm(self):
        return self._hmm
    
    @property
    def hmm(self):
        return self._hmm

    @hmm.setter
    def hmm(self, hmm):
        self._hmm = hmm
    
    @property
    def hidden_states(self):
        return self.hidden_to_external
        

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
        self._hmm = None
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

        if not set(start_probs.keys()).issubset(set(self.hidden_to_external)):
            raise InvalidInputError("start_prob keys match with transition keys")

        # Setup observed_to_internal, observed_to_external, and
        # num_observed_states
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
                raise InvalidInputError("You shouldn't see this")  # pragma: no cover
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

        self._hmm = HMM(
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
        return self.hidden_to_external

    def get_observable_states(self):
        """
        Returns:
            set: observable states
        """
        return self.observed_to_external

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
        internal_hidden = self._hmm.generate_hidden(time_steps)
        return [self.hidden_to_external[h] for h in internal_hidden]

    def generate_hidden_until_state(self, h):
        """
        Generates a sequence of hidden variables which stops at the first time step where we reach hidden state h.

        Parameters:
            h: stopping hidden state:

        Returns:
            list: Feasible sequence of hidden states, where last value is h
        """
        internal_hidden = self._hmm.generate_hidden_until_state(
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
        internal_observed = self._hmm.generate_observed_from_hidden(
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
        internal_observed = self._hmm.generate_observed(time_steps)
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
