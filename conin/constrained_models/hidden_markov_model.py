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
        ``"pyomo"``, ``"smoek"``, ``"toulbar2"``, ``"mvr"`` or
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
    * ``"variable_elimination"`` (no constraints, ``"factor_oracle"`` or
      ``"smoek"``; structure params ``start``, ``stop``): the factors of
      the unrolled model, used by variable elimination;
    * ``"toulbar2"`` (no constraints, ``"toulbar2"`` or ``"smoek"``;
      structure params ``start``, ``stop``): the cost function network;
    * ``"pyomo"`` (no constraints, ``"pyomo"`` or ``"smoek"``;
      structure params ``start``, ``stop``): the integer program of the
      unrolled model (``ip_formulation="markov_network"``);
    * ``"pyomo_network_flow"`` (no constraints, ``"pyomo"`` or
      ``"smoek"``): the network-flow integer program;
    * ``"mvr"`` (no constraints or ``"mvr"``): the built MVRs, aligned to
      the HMM (``representation("mvr").mvrs``), used by the torch MVR
      algorithms (:mod:`conin.algorithms.inference.mvr`, :mod:`conin.algorithms.learning.mvr`
      and :mod:`conin.algorithms.sampling.mvr`).
    """

    __slots__ = ()

    model_type = HiddenMarkovModel
