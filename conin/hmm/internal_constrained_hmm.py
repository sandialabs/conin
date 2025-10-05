from conin.exceptions import InvalidInputError
from conin.constraint import Constraint

# from conin.hmm import HMM


class Internal_Oracle_CHMM:
    def __init__(self, *, internal_hmm, constraints=None):
        """
        Constructs all the necessary attributes for the ConstrainedHMM object.

        Parameters:
            internal_hmm (HMM): An instance of the HMM class (default is None, which initializes a new HMM instance).
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
