# import copy
from conin.exceptions import InvalidInputError
from conin.constraint import Constraint
from conin.hmm import HMM
from . import internal_constrained_hmm
from . import chmm_base

# TODO I Really don't like this initalization and load logic


class Oracle_CHMM(chmm_base.CHMM_Base):
    """
    A class to represent a Hidden Markov Model (HMM) with additional constraints.
    """

    def __init__(self, *, hmm=None, constraints=None):
        """
        Constructs all the necessary attributes for the ConstrainedHMM object.

        Parameters:
            hmm (HMM, optional): An instance of the HMM class (default is None, which initializes a new HMM instance).
            constraints (list, optional): A list of constraints to be applied to the HMM (default is an empty list).
        """
        super().__init__(hmm=hmm)

        if constraints is None:
            self.constraints = []
        else:
            self.constraints = [con for con in constraints]

        self.load_internal_constrained_hmm()

    def get_constrained_hmm(self):
        return self

    def load_model(
        self, *, start_probs=None, transition_probs=None, emission_probs=None, hmm=None
    ):
        super().load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
            hmm=hmm,
        )
        self.load_internal_constrained_hmm()

    def load_internal_constrained_hmm(self):
        """
        Creates self.internal_constrained_hmm
        """
        # Make internal constraints
        internal_constraints = []
        for constraint in self.constraints:
            internal_constraints.append(self.make_internal_constraint(constraint))

        self.internal_constrained_hmm = internal_constrained_hmm.Internal_Oracle_CHMM(
            internal_hmm=self.hmm.internal_hmm,
            constraints=internal_constraints,
        )

    def make_internal_constraint(self, constraint):
        """
        Makes an internal version of the constraint that works on indices rather than keys

        Parameters:
            constraint (Constraint): The constraint we wish to make internal

        Returns:
            Constraint: An internalized version of constraint
        """
        if self.hmm is not None:

            def internal_func(internal_seq):
                external_seq = [self.hmm.hidden_to_external[h] for h in internal_seq]
                return constraint(external_seq)

            def internal_partial_func(T, internal_seq):
                external_seq = [self.hmm.hidden_to_external[h] for h in internal_seq]
                return constraint.partial_func(T, external_seq)

            internal_constraint = Constraint(
                func=internal_func,
                name="internal_" + constraint.name,
                partial_func=internal_partial_func,
            )

            return internal_constraint

    def add_constraint(self, constraint):
        """
        Adds a new constraint to the HMM.
        Automatically updates internal constraints if possible.

        Parameters:
            constraint (Constraint): A function that takes a sequence as input and returns a boolean indicating whether the sequence satisfies the constraint.
        """
        self.constraints.append(constraint)
        self.internal_constrained_hmm.add_constraint(
            self.make_internal_constraint(constraint)
        )

    def set_constraints(self, constraints):
        """
        Resets the constraints
        Automatically updates internal constraints if possible.

        Parameters:
            constraint (array of Constraint): Our new Constraints
        """
        self.constraints = []
        for constraint in constraints:
            self.add_constraint(constraint)
        self.load_internal_constrained_hmm()

    def generate_hidden(self, time_steps):
        """
        Generates a sequence of hidden variables satisfying the internal constraints

        Parameters:
            time_steps (int): How long you want the sequence to be

        Returns:
            list: Feasible sequence of hidden indices

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        internal_hidden = self.internal_constrained_hmm.generate_hidden(time_steps)
        return [self.hmm.hidden_to_external[h] for h in internal_hidden]

    def generate_observed_from_hidden(self, hidden):
        """
        Generates random observed sequence of states from a hidden sequence of states.

        Parameters:
            hidden (list): What we wish to generate from

        Returns:
            list: Observations generated from hidden
        """
        internal_hidden = [self.hmm.hidden_to_internal[h] for h in hidden]
        internal_observed = self.internal_constrained_hmm.generate_observed_from_hidden(
            internal_hidden
        )
        return [self.hmm.observed_to_external[o] for o in internal_observed]

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
            raise InvalidInputError("In generate_observed time_steps > 0.")
        internal_observed = self.internal_constrained_hmm.generate_observed(time_steps)
        return [self.hmm.observed_to_external[o] for o in internal_observed]

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
        for constraint in self.constraints:
            if not constraint.partial_func(T, seq):
                return False
        return True
