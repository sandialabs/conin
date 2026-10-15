from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel

from .base import ConstrainedGraphicalModel


class ConstrainedHiddenMarkovModel(ConstrainedGraphicalModel):
    """Hidden Markov model paired with constraints.

    Parameters
    ----------
    model : HiddenMarkovModel
        Hidden Markov model to wrap.
    constraints : Constraint or iterable of Constraint, optional
        Constraints of one family: ``"oracle"``, ``"factor_oracle"``,
        ``"pyomo"``, ``"algebraic"``, ``"toulbar2"`` or ``"mvr"``.
    data : object, optional
        Application data passed to the representation builders (MVR
        factories, Pyomo constraint functions, oracle wrappers).

    Notes
    -----
    Registered representations (see :meth:`representation`):

    * ``"oracle"`` (no constraints or ``"oracle"``): sampling and feasibility
      checks on internal hidden-state indices, used by A*;
    * ``"pyomo"`` (no constraints, ``"pyomo"`` or ``"algebraic"``;
      structure params ``start``, ``stop``): the integer program of the
      unrolled model (``ip_formulation="markov_network"``);
    * ``"pyomo_network_flow"`` (no constraints, ``"pyomo"`` or
      ``"algebraic"``): the network-flow integer program;
    * ``"mvr"`` (``"mvr"``): the built MVRs, aligned to the HMM
      (``representation("mvr").mvrs``).
    """

    __slots__ = ()

    model_type = HiddenMarkovModel
    supported_families = (
        None,
        "oracle",
        "factor_oracle",
        "pyomo",
        "algebraic",
        "toulbar2",
        "mvr",
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
            self.model.hidden_to_external[h]
            for h in self.representation("oracle").generate_hidden(time_steps)
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
        internal_hidden = [self.model.hidden_to_internal[h] for h in hidden]
        if not self.representation("oracle").is_feasible(internal_hidden):
            raise InvalidInputError(
                "ConstrainedHiddenMarkovModel.generate_observed_from_hidden() - The sequence of hidden states is not feasible."
            )
        internal_observed = self.model.repn.generate_observed_from_hidden(
            internal_hidden
        )
        return [self.model.observed_to_external[o] for o in internal_observed]

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
        states = {i: v for i, v in enumerate(seq)}
        for constraint in self.constraints:
            if not constraint(states):
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
        return self.representation("oracle").partial_is_feasible(
            T=T, seq=[self.model.hidden_to_internal[h] for h in seq]
        )
