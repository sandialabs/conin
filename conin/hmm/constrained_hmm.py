from conin.constraint import OracleConstraint, PyomoConstraint
from conin.exceptions import InvalidInputError
from .chmm import CHMM


class ConstrainedHiddenMarkovModel:
    """
    A class to represent a base Hidden Markov Model (HMM).
    """

    def __init__(self, *, hmm=None, constraints=None):
        """
        Constructor.

        Parameters:
            hmm (HiddenMarkovModel or HMM, optional): An instance of the HiddenMarkovModel
            or HMM class (default is None).
        """
        self.hidden_markov_model = hmm
        self.constraint_type = None
        if constraints:
            self.constraints = constraints
        else:
            self._constraints = []

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
        if isinstance(constraint, OracleConstraint):
            assert self.constraint_type is None or self.constraint_type == "oracle"
            self.constraint_type = "oracle"
            self._constraints.append(constraint)
        elif isinstance(constraint, PyomoConstraint):
            assert self.constraint_type is None or self.constraint_type == "pyomo"
            self.constraint_type = "pyomo"
            self._constraints.append(constraint)
        else:
            raise ValueError(f"Unexpected constraint type: {type(constraint)=}")

    def initialize_chmm(self, constraint_type=None, *, data=None):
        if constraint_type:
            self.constraint_type = constraint_type
        if self.constraint_type is None:
            self.chmm = CHMM(
                hmm=self.hidden_markov_model.repn, constraints=self.constraints
            )
        elif self.constraint_type == "oracle":
            from .chmm_oracle import Oracle_CHMM

            self.chmm = Oracle_CHMM(
                hmm=self.hidden_markov_model.repn,  # HMM object
                constraints=self.constraints,  # list of OracleConstraint objects
                hidden_to_external=self.hidden_markov_model.hidden_to_external,
                data=data,  # Application-specific data
            )
        elif self.constraint_type == "pyomo":
            from .chmm_algebraic import PyomoAlgebraic_CHMM

            self.chmm = PyomoAlgebraic_CHMM(
                hidden_markov_model=self.hidden_markov_model,  # HiddenMarkovModel object
                constraints=self.constraints,  # list of PyomoConstraint objects
                data=data,  # Application-specific data
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
        internal_observed = self.hidden_markov_model.repn.generate_observed_from_hidden(
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
        for constraint in self.constraints:
            if not constraint(seq):
                return False
        return True

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
