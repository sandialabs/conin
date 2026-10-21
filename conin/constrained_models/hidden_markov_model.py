from conin.graphical_models import HiddenMarkovModel

from .base import ConstrainedGraphicalModel
from .feasibility import (
    check_constraints,
    hmm_constraint_data,
    hmm_sequences,
    hmm_state_space,
)


class ConstrainedHiddenMarkovModel(ConstrainedGraphicalModel):
    """Hidden Markov model paired with constraints.

    Parameters
    ----------
    model : HiddenMarkovModel
        Hidden Markov model to wrap.
    constraints : Constraint or iterable of Constraint, optional
        Constraints of one family: ``"oracle"``, ``"factor_oracle"``,
        ``"pyomo"``, ``"algebraic"``, ``"toulbar2"``, ``"mvr"`` or
        ``"recursive"``.
    data : object, optional
        Application data passed to the representation builders (MVR
        factories, Pyomo constraint functions, oracle wrappers).

    Notes
    -----
    Registered representations (see :meth:`representation`):

    * ``"matvec"`` (no constraints only): the HMM parameters as read-only
      arrays on internal indices, used by Viterbi, A*, sampling and the torch
      MVR kernels;
    * ``"oracle"`` (no constraints or ``"oracle"``): feasibility checks on
      internal hidden-state indices, used by A* and rejection sampling;
    * ``"recursive"`` (no constraints or ``"recursive"``): the combined
      recursive constraint, used by recursive A*;
    * ``"factor"`` (no constraints, ``"factor_oracle"`` or ``"algebraic"``;
      structure params ``start``, ``stop``): variable elimination;
    * ``"toulbar2"`` (no constraints, ``"toulbar2"`` or ``"algebraic"``;
      structure params ``start``, ``stop``): the cost function network;
    * ``"pyomo"`` (no constraints, ``"pyomo"`` or ``"algebraic"``;
      structure params ``start``, ``stop``): the integer program of the
      unrolled model (``ip_formulation="markov_network"``);
    * ``"pyomo_network_flow"`` (no constraints, ``"pyomo"`` or
      ``"algebraic"``): the network-flow integer program;
    * ``"mvr"`` (no constraints or ``"mvr"``): the built MVRs, aligned to
      the HMM (``representation("mvr").mvrs``), used by the torch MVR
      algorithms (:mod:`conin.inference.mvr`, :mod:`conin.learning.mvr`,
      :mod:`conin.sampling.mvr`, :mod:`conin.queries.mvr`).
    """

    __slots__ = ()

    model_type = HiddenMarkovModel

    def is_feasible(self, seq, observed=None):
        """Check whether a hidden-state sequence satisfies all configured constraints.

        Parameters
        ----------
        seq : list or dict
            Sequence of external hidden-state labels (a list, or a dict keyed
            by the times ``0..T-1``).
        observed : list or dict, optional
            The observed labels, needed only by ``"factor_oracle"`` and
            ``"algebraic"`` constraints that refer to the ``("E", t)`` nodes.

        Returns
        -------
        bool
            ``True`` if every constraint is satisfied and ``False`` otherwise.

            * ``"oracle"`` predicates see ``{t: hidden}``;
            * ``"factor_oracle"`` and ``"algebraic"`` constraints see the
              nodes ``("H", t)`` (and ``("E", t)``) of the unrolled model, as
              for variable elimination; ``"algebraic"`` needs smoek;
            * ``"mvr"`` constraints are run along the sequence (see
              :func:`~conin.constrained_models.feasibility.mvr_accepts`).

            Constraint functions that take a second argument see
            ``data = Munch(hmm=Munch(T=list(range(T))))``.  ``MVRConstraint``
            factories are called with ``(hmm, self.data)``.

        Raises
        ------
        InvalidInputError
            For an empty or badly keyed sequence, unknown labels, or a
            constraint that needs observations that were not given.
        NotImplementedError
            For ``"pyomo"`` and ``"toulbar2"`` constraints, whose functions can
            add variables to the backend model.
        """
        hidden, observed = hmm_sequences(self.model, seq, observed)
        T = len(hidden)
        factor_assignment = {("H", t): h for t, h in enumerate(hidden)}
        if observed is not None:
            factor_assignment.update({("E", t): o for t, o in enumerate(observed)})
        return check_constraints(
            self,
            assignment=dict(enumerate(hidden)),
            factor_assignment=factor_assignment,
            state_space=hmm_state_space(self.model, T),
            data=hmm_constraint_data(T),
            hidden=hidden,
        )

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
