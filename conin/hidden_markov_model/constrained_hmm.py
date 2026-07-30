from conin.constraint import (
    OracleConstraint,
    PyomoConstraint,
    Toulbar2Constraint,
    FactorConstraint,
)
from conin.exceptions import InvalidInputError
from .chmm import CHMM


class ConstrainedHiddenMarkovModel:
    """Hidden Markov model wrapper with attached constraint functors.

    Parameters
    ----------
    hmm : HiddenMarkovModel or HMM_MatVecRepn, optional
        Hidden Markov model to wrap.
    constraints : list[ConstraintFunctor], optional
        Constraint functors used to build a constrained HMM solver.
    """

    def __init__(self, *, hmm=None, constraints=None):
        """Initialize a constrained hidden Markov model wrapper.

        Parameters
        ----------
        hmm : HiddenMarkovModel or HMM_MatVecRepn, optional
            Hidden Markov model to wrap.
        constraints : list[ConstraintFunctor], optional
            Constraint functors used to configure the constrained model.
        """
        self.hidden_markov_model = hmm
        self.constraint_type = None
        if constraints:
            self.constraints = constraints
        else:
            self._constraints = []

    @property
    def constraints(self):
        """Get the configured constraint functors.

        Returns
        -------
        list
            List of ``ConstraintFunctor`` instances.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_list):
        """Set the list of constraint functors used to define model constraints.

        Parameters
        ----------
        constraint_list : list[ConstraintFunctor]
            List of ``ConstraintFunctor`` instances that generate model
            constraints.
        """
        assert type(constraint_list) is list
        self._constraints = []
        for con in constraint_list:
            self.add_constraint(con)

    def add_constraint(self, constraint):
        """Add a single constraint functor to the model.

        Parameters
        ----------
        constraint : ConstraintFunctor
            Constraint functor to register.

        Raises
        ------
        ValueError
            If ``constraint`` is not a supported constraint functor type.
        """
        if isinstance(constraint, OracleConstraint):
            assert self.constraint_type is None or self.constraint_type == "oracle"
            self.constraint_type = "oracle"
            self._constraints.append(constraint)
        elif isinstance(constraint, PyomoConstraint):
            assert self.constraint_type is None or self.constraint_type == "pyomo"
            self.constraint_type = "pyomo"
            self._constraints.append(constraint)
        elif isinstance(constraint, Toulbar2Constraint):
            assert self.constraint_type is None or self.constraint_type == "toulbar2"
            self.constraint_type = "toulbar2"
            self._constraints.append(constraint)
        elif isinstance(constraint, FactorConstraint):
            assert self.constraint_type is None or self.constraint_type == "factor"
            self.constraint_type = "factor"
            self._constraints.append(constraint)
        else:
            raise ValueError(f"Unexpected constraint type: {type(constraint)=}")

    def initialize_chmm(self, constraint_type=None, *, data=None, **kwargs):
        """Initialize the internal constrained HMM solver.

        Parameters
        ----------
        constraint_type : {"oracle", "pyomo", "toulbar2", "factor"}, optional
            Explicit constraint backend to use.
        data : optional
            Application-specific data passed to the solver.
        **kwargs
            Additional keyword arguments forwarded to the selected solver
            implementation.
        """
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
                **kwargs,
            )
        elif self.constraint_type == "pyomo":
            from .chmm_algebraic import PyomoAlgebraic_CHMM

            self.chmm = PyomoAlgebraic_CHMM(
                hidden_markov_model=self.hidden_markov_model,  # HiddenMarkovModel object
                constraints=self.constraints,  # list of PyomoConstraint objects
                data=data,  # Application-specific data
                **kwargs,
            )

    def generate_hidden(self, time_steps):
        """Generate a feasible sequence of hidden states.

        Parameters
        ----------
        time_steps : int
            Number of hidden states to generate.

        Returns
        -------
        list
            Feasible sequence of external hidden-state labels.
        """
        return [
            self.hidden_markov_model.hidden_to_external[h]
            for h in self.chmm.generate_hidden(time_steps)
        ]

    def generate_observed_from_hidden(self, hidden):
        """Generate observed states from a feasible hidden-state sequence.

        Parameters
        ----------
        hidden : list
            External hidden-state labels from which to generate observations.

        Returns
        -------
        list
            Observed-state labels sampled from the wrapped hidden Markov model.

        Raises
        ------
        InvalidInputError
            If ``hidden`` does not satisfy the configured constraints.
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
        """Generate a feasible hidden-state sequence and sample observations.

        Parameters
        ----------
        time_steps : int
            Number of time steps to generate.

        Returns
        -------
        list
            Observed-state labels sampled from a feasible hidden-state sequence.

        Raises
        ------
        InvalidInputError
            If ``time_steps`` is negative.
        """
        if time_steps < 0:
            raise InvalidInputError("In generate_observed, time_steps must be >= 0.")

        hidden = self.generate_hidden(time_steps)
        return self.generate_observed_from_hidden(hidden)

    def is_feasible(self, seq):
        """Check whether a sequence satisfies all configured constraints.

        Parameters
        ----------
        seq : list
            Sequence of external hidden-state labels.

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
            Target full sequence length.
        seq : list
            Partial sequence of external hidden-state labels.

        Returns
        -------
        bool
            ``True`` if the partial sequence admits a feasible completion and
            ``False`` otherwise.
        """
        return self.chmm.partial_is_feasible(
            T=T, seq=[self.hidden_markov_model.hidden_to_internal[h] for h in seq]
        )
