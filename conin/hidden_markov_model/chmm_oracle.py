from conin.constraint import OracleConstraint

from . import chmm


class Oracle_CHMM(chmm.CHMM):
    """Constrained HMM that enforces oracle-style sequence constraints."""

    def __init__(
        self,
        *,
        hmm=None,
        constraints=None,
        hidden_to_external={},
        data=None,
        make_internal_constraint=True,
    ):
        """Initialize an oracle-constrained HMM.

        Parameters
        ----------
        hmm : HMM_MatVecRepn, optional
            Numeric hidden Markov model representation used for sampling.
        constraints : list of OracleConstraint, optional
            Oracle constraints to apply to generated hidden-state sequences.
        hidden_to_external : dict, optional
            Mapping from internal hidden-state indices to external labels.
        data : optional
            Application-specific data passed through to the base constrained
            HMM.
        make_internal_constraint : bool, optional
            If ``True``, wrap each oracle constraint so it operates on internal
            hidden-state indices.
        """
        super().__init__(hmm=hmm, data=data)
        if constraints:
            if make_internal_constraint:
                self.constraints = [
                    self._make_internal_constraint(c, hidden_to_external)
                    for c in constraints
                ]
            else:
                self.constraints = constraints
        else:
            self.constraints = []

    def _make_internal_constraint(self, constraint, hidden_to_external):
        """Convert an external oracle constraint to internal index space.

        Parameters
        ----------
        constraint : OracleConstraint
            Constraint defined on external hidden-state labels.
        hidden_to_external : dict
            Mapping from internal hidden-state indices to external labels.

        Returns
        -------
        OracleConstraint
            Constraint that accepts internal hidden-state indices.
        """

        def internal_func(internal_seq):
            external_seq = [hidden_to_external[h] for h in internal_seq]
            return constraint(external_seq)

        def internal_partial_func(T, internal_seq):
            external_seq = [hidden_to_external[h] for h in internal_seq]
            return constraint.partial_func(T, external_seq)

        internal_constraint = OracleConstraint(
            func=internal_func,
            name="internal_" + constraint.name,
            partial_func=internal_partial_func,
        )

        return internal_constraint

    def generate_hidden(self, time_steps, max_failures=1000):
        """Generate a hidden-state sequence satisfying all oracle constraints.

        Parameters
        ----------
        time_steps : int
            Number of hidden states to generate.
        max_failures : int, optional
            Maximum number of rejected samples before raising an error.

        Returns
        -------
        list
            Feasible sequence of internal hidden-state indices.

        Raises
        ------
        RuntimeError
            If no feasible sequence is found within ``max_failures`` trials.
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
        """Check whether a sequence satisfies all oracle constraints.

        Parameters
        ----------
        seq : list
            Sequence of internal hidden-state indices.

        Returns
        -------
        bool
            ``True`` if every constraint is satisfied and ``False`` otherwise.
        """
        for constraint in self.constraints:
            if not constraint(seq):
                return False
        return True

    def partial_is_feasible(self, *, T, seq):
        """Check whether a partial sequence can still be extended feasibly.

        Parameters
        ----------
        T : int
            Target sequence length.
        seq : list
            Partial sequence of internal hidden-state indices.

        Returns
        -------
        bool
            ``True`` if every constraint allows some feasible completion of the
            partial sequence and ``False`` otherwise.
        """
        for constraint in self.constraints:
            if not constraint.partial_func(T, seq):
                return False
        return True
