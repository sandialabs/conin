"""The ``"oracle"`` representation of a constrained HMM.

It replaces ``Oracle_CHMM``: the feasibility checks used by A*
(:func:`conin.inference.oracle.a_star`) and by rejection sampling
(:func:`conin.sampling.oracle.rejection_sample_hidden`), on sequences of
internal hidden-state indices.
"""

from conin.constrained_models.feasibility import hmm_constraint_data, oracle_accepts
from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation
from conin.representations.matvec import MatVecHMMRepresentation
from conin.util.constraint_data import merge_constraint_data


class OracleHMMRepresentation(Representation):
    """Oracle view of a constrained HMM: feasibility checks on internal indices.

    The oracle constraints are written against dicts of external labels keyed
    by time (``{0: "h0", 1: "h1", ...}``); this representation translates the
    internal-index sequences of the search and sampling code to that form.

    Attributes
    ----------
    model : HiddenMarkovModel
        The wrapped HMM.
    hmm : MatVecHMMRepresentation
        The HMM parameters as arrays (built from ``model``, as
        ``representation("matvec")`` of the bare HMM), used by A* and sampling.
    constraints : tuple of OracleConstraint
        The constraints of the constrained model (not copied or wrapped).
    data : object or None
        The constrained model's application data.

    Notes
    -----
    Nothing here changes after the build, so one representation can serve
    several searches or samples.  A predicate ``func(states, data)`` gets
    a copy of the application data with ``hmm = Munch(T=list(range(T)))`` set
    (``Munch(hmm=Munch(T=...))`` if there is none) for a sequence of length
    ``T``, as in ``ConstrainedHiddenMarkovModel.is_feasible``.
    """

    kind = "oracle"

    def __init__(self, cmodel):
        self.model = cmodel.model
        self.hmm = MatVecHMMRepresentation(self.model)
        self.constraints = cmodel.constraints
        self.data = cmodel.data
        self._hidden_to_external = self.model.hidden_to_external

    def _external(self, seq):
        h2e = self._hidden_to_external
        return {t: h2e[h] for t, h in enumerate(seq)}

    def is_feasible(self, seq):
        """Check a full sequence of internal hidden-state indices.

        Parameters
        ----------
        seq : list or tuple of int

        Returns
        -------
        bool
            ``True`` if every constraint's predicate accepts the sequence.
        """
        if not self.constraints:
            return True
        states = self._external(seq)
        data = merge_constraint_data(
            self.data, hmm_constraint_data(self.model, len(seq))
        )
        return all(oracle_accepts(c, states, data) for c in self.constraints)

    def partial_is_feasible(self, *, T, seq):
        """Check whether a prefix of internal indices may still be completed.

        Parameters
        ----------
        T : int
            Length of the full sequence.
        seq : list or tuple of int
            The prefix.

        Returns
        -------
        bool
            ``True`` if every constraint's ``partial_func(T, states)`` accepts
            the prefix.
        """
        if not self.constraints:
            return True
        states = self._external(seq)
        return all(c.partial_func(T, states) for c in self.constraints)


@register_representation("oracle", HiddenMarkovModel, families=(None, "oracle"))
def build_oracle_hmm_representation(cmodel):
    return OracleHMMRepresentation(cmodel)
