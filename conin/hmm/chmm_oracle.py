from conin.constraint import Constraint

from . import chmm


class Oracle_CHMM(chmm.CHMM):
    """
    A class to represent a Hidden Markov Model (HMM) with additional constraints.
    """

    def __init__(self, *, hmm=None, constraints=None, hidden_to_external={}, data=None):
        """
        Parameters:
            hmm (HMM, optional):
                An instance of the HMM class (default is None,
                which initializes a new HMM instance).
            constraints (list, optional):
                A list of constraintsto be applied to the HMM (default is an empty list).
        """
        super().__init__(hmm=hmm, data=data)
        if constraints:
            self.constraints = [
                self._make_internal_constraint(c, hidden_to_external)
                for c in constraints
            ]
        else:
            self.constraints = []

    def _make_internal_constraint(self, constraint, hidden_to_external):
        """
        Makes an internal version of the constraint that works on indices rather than keys

        Parameters:
            constraint (Constraint): The constraint we wish to make internal

        Returns:
            Constraint: An internalized version of constraint
        """

        def internal_func(internal_seq):
            external_seq = [hidden_to_external[h] for h in internal_seq]
            return constraint(external_seq)

        def internal_partial_func(T, internal_seq):
            external_seq = [hidden_to_external[h] for h in internal_seq]
            return constraint.partial_func(T, external_seq)

        internal_constraint = Constraint(
            func=internal_func,
            name="internal_" + constraint.name,
            partial_func=internal_partial_func,
        )

        return internal_constraint

    def generate_hidden(self, time_steps, max_failures=1000):
        """
        Generates a sequence of hidden variables satisfying the internal constraints

        Parameters:
            time_steps (int): How long you want the sequence to be

        Returns:
            list: Feasible sequence of hidden indices

        Raises:
            InvalidInputError: If time_steps is negative.
        """
        hidden = self.hmm.generate_hidden(time_steps)
        ctr = 0
        while not self.is_feasible(hidden):
            hidden = self.hmm.generate_hidden(time_steps)
            ctr += 1
            if ctr > max_failures:
                raise RuntimeError(
                    f"Failed to generate a feasible hidden state after {max_failures} trials"
                )
        return hidden

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
