from clio import InvalidInputError
from clio import Constraint
from clio.hmm import Internal_HMM


class Internal_Oracle_CHMM:
    def __init__(self, *, internal_hmm, constraints=None):
        """
        Constructs all the necessary attributes for the ConstrainedHMM object.

        Parameters:
            internal_hmm (Internal_HMM): An instance of the Internal_HMM class (default is None, which initializes a new HMM instance).
            constraints (list, optional): A list of constraints to be applied to the HMM (default is an empty list).
        """
        self.internal_hmm = internal_hmm

        self.constraints = []
        if constraints is not None:
            self.set_constraints(constraints)

    def add_constraint(self, constraint):
        """
        Adds a new constraint to the HMM.

        Parameters:
            constraint (Constraint): A function that takes a sequence as input and returns a boolean indicating whether the sequence satisfies the constraint.
        """

        self.constraints.append(constraint)

    def set_constraints(self, constraints):
        """
        As opposed to add constraint, this one first clears constraints, and then adds them

        Parameters:
            constraints (array of Constraint): Constratins which we wish to set with
        """
        self.constraints = []
        for constraint in constraints:
            self.add_constraint(constraint)

    def get_internal_hmm(self):
        """
        Returns internal_hmm
        """
        return self.internal_hmm

    def is_feasible(self, seq):
        """
        Check if an sequence satisfies the constraints

        Parameters:
            seq (list): A sequence to be checked against theconstraints.

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
        for constraint in self.constraints:
            if not constraint.partial_func(T, seq):
                return False
        return True

    def generate_hidden(self, time_steps):
        """
        Generates random series of indices satisfying the constraints

        Parameters:
            time_steps (int): How long you want the sequence to be

        Returns:
            list: Feasible sequence of hidden indices

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_hidden, time_steps >= 0.")

        while True:
            hidden = self.internal_hmm.generate_hidden(time_steps)
            if self.is_feasible(hidden):
                return hidden

    def generate_observed_from_hidden(self, hidden):
        """
        Generates random observed sequence of states from a hidden sequence of states.

        Parameters:
            hidden (list): What we wish to generate from

        Returns:
            list: Observations generated from hidden

        Raises:
            InvalidInputError: if hidden is not feasible
        """
        if not self.is_feasible(hidden):
            raise InvalidInputError(
                "In generate_observed_from_hidden the hidden sequences of states must satisfy the constraints"
            )
        return self.internal_hmm.generate_observed_from_hidden(hidden)

    def generate_observed(self, time_steps):
        """
        Generates random observed sequence of states.

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
        return self.internal_hmm.generate_observed_from_hidden(hidden)
