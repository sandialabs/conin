import numpy as np
from conin import InvalidInputError
from conin.util import Util
from conin.hmm import Internal_Statistical_Model


class Internal_HMM(Internal_Statistical_Model):
    def __init__(self, *, start_vec, transition_mat, emission_mat):
        self.load_start_vec(start_vec)
        self.load_transition_mat(transition_mat)
        self.load_emission_mat(emission_mat)
        self.check_dimensions()
        self.load_dimensions()

    def load_start_vec(self, start_vec):
        """
        Loads the start vec

        Parameters:
            start_vec (array): Starting Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the probabilities do not sum to 1
        """
        for prob in start_vec:
            # Non-negative
            if not prob >= 0:
                raise InvalidInputError("start_probs values must be positive floats.")
        # Sums to 1
        if not np.isclose(sum(start_vec), 1):
            raise InvalidInputError("start_prob values must sum to 1.")
        self.start_vec = start_vec

    def load_transition_mat(self, transition_mat):
        """
        Loads the transition matrix

        Parameters:
            transition_mat (array): Transition Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the rows do not sum to 1
        """

        for h1 in range(len(transition_mat)):
            for h2 in range(len(transition_mat[h1])):
                if not transition_mat[h1][h2] >= 0:
                    raise InvalidInputError("Transition_mat must be positive floats.")
        # Rows sum to 1
        for vec in transition_mat:
            if not np.isclose(sum(vec), 1):
                raise InvalidInputError("Transition_mat rows do not sum to 1.")
        self.transition_mat = transition_mat

    def load_emission_mat(self, emission_mat):
        """
        Loads the emission matrix

        Parameters:
            emission_mat (array): Emission Probabilities

        Raises:
            InvalidInputError: If any probabilities are negative or if the rows do not sum to 1
        """

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
        Checks that the dimensions of the input matrices are appropriately sized

        Raises:
            InvalidInputError: If any dimensions do not line up correctly
        """
        correct_dimension = True

        if len(self.start_vec) != len(self.transition_mat):
            correct_dimension = False
        for vec in self.transition_mat:
            if len(self.start_vec) != len(vec):
                correct_dimension = False

        if len(self.start_vec) != len(self.emission_mat):
            correct_dimension = False
        for vec in self.emission_mat:
            if len(self.emission_mat[0]) != len(vec):
                correct_dimension = False

        if not correct_dimension:
            raise InvalidInputError(
                "Dimensions do not line up correctly in check_dimensions"
            )

    def load_dimensions(self):
        """
        Updates num_hidden and num_observed
        """
        self.num_hidden_states = len(self.start_vec)
        self.num_observed_states = len(self.emission_mat[0])
        self.hidden_states = range(self.num_hidden_states)
        self.observed_states = range(self.num_observed_states)

    def get_internal_hmm(self):
        """
        Returns internal_hmm
        """
        return self

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
            raise InvalidInputError("In generate_observed time_steps > 0.")
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
