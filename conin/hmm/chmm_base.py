from conin.exceptions import InvalidInputError
from conin.hmm import HMM, Statistical_Model


class CHMM_Base(Statistical_Model):
    """
    A class to represent a base Hidden Markov Model (HMM).
    """

    def __init__(self, *, hmm=None):
        """
        Constructor.

        Parameters:
            hmm (HMM, optional): An instance of the HMM class (default is None, which initializes a new HMM instance).
        """
        self.hmm = HMM() if hmm is None else hmm

    def get_hmm(self):
        """
        Returns hmm
        """
        return self.hmm

    def get_internal_hmm(self):
        """
        Returns internal hmm
        """
        return self.hmm.internal_hmm

    def load_model(
            self,
            *,
            start_probs=None,
            transition_probs=None,
            emission_probs=None,
            hmm=None):
        """
        Loads the HMM model with the given parameters.
        Either give all three dictionaries or hmm, but not both

        Parameters:
            start_probs (dict, optional): A dictionary representing the start probabilities.
            transition_probs (dict, optional): A dictionary representing the transition probabilities.
            emission_probs (dict, optional: A dictionary representing the emission probabilities.
            hmm (HMM, optional): The HMM we wish to load with.

        Raises:
            InvalidInputError: If we supply too much or not enough information
        """
        if (
            hmm is not None
            and start_probs is None
            and transition_probs is None
            and emission_probs is None
        ):
            # If an HMM object is provided, load it directly
            self.hmm = hmm
        elif (
            start_probs is not None
            and transition_probs is not None
            and emission_probs is not None
        ):
            # If dictionaries are provided, create an HMM object and load it
            hmm = HMM()
            hmm.load_model(
                start_probs=start_probs,
                transition_probs=transition_probs,
                emission_probs=emission_probs,
            )
            self.hmm = hmm
        else:
            raise InvalidInputError(
                "You must provide either an HMM object or all three dictionaries, and not both."
            )

    def is_valid_observed_state(self, o):
        """
        Check if the given observed state is allowed

        Parameters:
            o : Any: The observed state to be checked for validity.

        Returns:
            bool: True if the observed state `o` is valid (i.e., exists in the mapping),
            False otherwise.
        """
        return self.hmm.is_valid_observed_state(o)

    def is_valid_hidden_state(self, h):
        """
        Check if the given hidden state is allowed

        Parameters:
            h : Any: The hidden state to be checked for validity.

        Returns:
            bool: True if the hidden state `h` is valid (i.e., exists in the mapping),
            False otherwise.
        """
        return self.hmm.is_valid_hidden_state(h)

    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.

        Parameters:
            seed (int): The seed value to set for the random number generator.
        """
        self.hmm.set_seed(seed)

    def get_start_probs(self):
        """
        Retrieves the starting probabilities of the hidden states.

        Returns:
            dict: A dictionary mapping hidden states to their starting probabilities.
        """
        return self.hmm.get_start_probs()

    def get_transition_probs(self):
        """
        Retrieves the transition probabilities between hidden states.

        Returns:
            dict: A dictionary mapping pairs of hidden states to their transition probabilities.
        """
        return self.hmm.get_transition_probs()

    def get_emission_probs(self):
        """
        Retrieves the emission probabilities from hidden states to observed states.

        Returns:
            dict: A dictionary mapping pairs of hidden states and observed states to their emission probabilities.
        """
        self.hmm.get_emission_probs()

    def to_dict(self):
        """
        Generate a dict representation of the model data.

        Returns:
            dict: A dictionary representaiton of this statistical model.
        """
        return self.hmm.to_dict()

    def generate_hidden(self, time_steps):
        """
        Generates a feasible sequence of hidden states

        Parameters:
            time_steps (int): How long you want the sequence to be

        Returns:
            list: Feasible sequence of hidden indices

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        raise NotImplementedError(
            "CHMM_Base.generate_hidden() is not implemented")

    def generate_observed_from_hidden(self, hidden):
        """
        Generates random sequence of observed states from a sequence of hidden states.

        Parameters:
            hidden (list): What we wish to generate from

        Returns:
            list: Observations generated from hidden
        """
        if not self.is_feasible(hidden):
            raise InvalidInputError(
                "CHMM_Base.generate_observed_from_hidden() - The sequence of hidden states is not feasible."
            )
        internal_hidden = [self.hmm.hidden_to_internal[h] for h in hidden]
        return self.hmm.internal_hmm.generate_observed_from_hidden(hidden)

    def generate_observed(self, time_steps):
        """
        Generates random sequence of observed states.

        Parameters:
            hidden (list): What we wish to generate from

        Returns:
            list: Observations generated from hidden

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        if time_steps < 0:
            raise InvalidInputError(
                "In generate_observed, time_steps must be >= 0.")

        hidden = self.generate_hidden(time_steps)
        return self.generate_observed_from_hidden(hidden)

    def log_probability(self, observations, hidden):
        """
        Compute the log-probability of the observations given the hidden state.
        """
        raise NotImplementedError(
            "CHMM_Base.log_probability() is not implemented")

    def is_feasible(self, seq):
        """
        Checks if a given sequence satisfies all constraints.

        Parameters:
            seq (list): A sequence to be checked against the constraints.

        Returns:
            bool: True if the sequence satisfies all constraints, False otherwise.
        """
        raise NotImplementedError("CHMM_Base.is_feasible() is not implemented")
