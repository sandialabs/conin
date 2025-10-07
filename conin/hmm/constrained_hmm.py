from conin.constraint import Constraint, PyomoConstraint
from conin.exceptions import InvalidInputError
from conin.hmm import HiddenMarkovModel
from .chmm import CHMM
from .chmm_oracle import Oracle_CHMM


class ConstrainedHiddenMarkovModel:
    """
    A class to represent a base Hidden Markov Model (HMM).
    """

    def __init__(self, *, hmm=None, constraint_list=None):
        """
        Constructor.

        Parameters:
            hmm (HiddenMarkovModel or HMM, optional): An instance of the HiddenMarkovModel
            or HMM class (default is None).
        """
        self.hidden_markov_model = hmm
        if constraint_list:
            self.constraints = constraint_list
        else:
            self._constraints = []
        self.constraint_type = None

    @property
    def constraints(self):
        """Get a list of constraint functors.

        :return: The constraint functor or ``None`` if not set.
        :rtype: callable | None
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set a list of functions that are used to define model constraints.

        Parameters
        ----------
        constraint_list : List[Callable]
            List of functions that generate model constraints.
        """
        assert type(constraint_list) is list
        self._constraints = []
        for con in constraint_list:
            self.add_constraint(con)

    def add_constraint(self, constraint):
        if isinstance(constraint, Constraint):
            assert self.constraint_type is None or self.constraint_type == "oracle"
            self.constraint_type = "oracle"
            self._constraints.append(constraint)
        elif isinstance(constraint, PyomoConstraint):
            assert self.constraint_type is None or self.constraint_type == "pyomo"
            self.constraint_type = "pyomo"
            self._constraints.append(constraint)
        else:
            raise ValueError(f"Unexpected constraint type: {type(constraint)=}")

    def initialize_chmm(self):
        if self.constraint_type is None:
            self.chmm = CHMM(
                hmm=self.hidden_markov_model.hmm, constraints=self.constraints
            )
        elif self.constraint_type == "oracle":
            self.chmm = Oracle_CHMM(
                hmm=self.hidden_markov_model.hmm,
                constraints=self.constraints,
                hidden_to_external=self.hidden_markov_model.hidden_to_external,
            )
        elif self.constraint_type == "pyomo":
            self.chmm = Algebraic_CHMM(
                hmm=self.hidden_markov_model.hmm, constraints=self.constraints
            )

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
        return [
            self.hidden_markov_model.hidden_to_external[h]
            for h in self.chmm.generate_hidden(time_steps)
        ]

    def generate_observed_from_hidden(self, hidden):
        """
        Generates random sequence of observed states from a sequence of hidden states.

        Parameters:
            hidden (list): What we wish to generate from

        Returns:
            list: Observations generated from hidden
        """
        internal_hidden = [
            self.hidden_markov_model.hidden_to_internal[h] for h in hidden
        ]
        if not self.chmm.is_feasible(internal_hidden):
            raise InvalidInputError(
                "ConstrainedHiddenMarkovModel.generate_observed_from_hidden() - The sequence of hidden states is not feasible."
            )
        internal_observed = self.hidden_markov_model.hmm.generate_observed_from_hidden(
            internal_hidden
        )
        return [
            self.hidden_markov_model.observed_to_external[o] for o in internal_observed
        ]

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
            raise InvalidInputError("In generate_observed, time_steps must be >= 0.")

        hidden = self.generate_hidden(time_steps)
        return self.generate_observed_from_hidden(hidden)

    def is_feasible(self, seq):
        """
        Checks if a given sequence satisfies all constraints.

        Parameters:
            seq (list): A sequence to be checked against the constraints.

        Returns:
            bool: True if the sequence satisfies all constraints, False otherwise.
        """
        return self.chmm.is_feasible(
            [self.hidden_markov_model.hidden_to_internal[h] for h in seq]
        )

    def partial_is_feasible(self, *, T, seq):
        """
        Check if a partial sequence satisfies the constraints
        E.g. returns false only if there is no possibility of extending the sequence to
        being feasible.

        Parameters:
            seq (list): A sequence to be checked against the constraints.
        Returns:
            bool: True if the sequence satisfies all constraints, False otherwise.
        """
        return self.chmm.partial_is_feasible(
            T=T, seq=[self.hidden_markov_model.hidden_to_internal[h] for h in seq]
        )


class XConstrainedHiddenMarkovModel:

    def get_hmm(self):
        return self.hmm

    def get_internal_hmm(self):
        """
        Returns internal hmm
        """
        return self.hmm.internal_hmm

    def load_model(
        self, *, start_probs=None, transition_probs=None, emission_probs=None, hmm=None
    ):
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
            hmm = HiddenMarkovModel()
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
        self._seed = 1
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
        return self.hmm.get_emission_probs()

    def to_dict(self):
        """
        Generate a dict representation of the model data.

        Returns:
            dict: A dictionary representaiton of this statistical model.
        """
        raise NotImplementedError(
            "We could return the hmm dict, but that doesn't capture constraint info."
        )

    def log_probability(self, observations, hidden):
        """
        Compute the log-probability of the observations given the hidden state.
        """
        raise NotImplementedError(
            "ConstrainedHiddenMarkovModel.log_probability() is not implemented"
        )
