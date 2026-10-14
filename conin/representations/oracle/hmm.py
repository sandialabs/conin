"""The ``"oracle"`` representation of a constrained HMM.

It wraps :class:`conin.hidden_markov_model.chmm_oracle.Oracle_CHMM`, built the
way ``ConstrainedHiddenMarkovModel.initialize_chmm()`` built it before PR 3.
``Oracle_CHMM`` becomes this representation in PR 4c.
"""

from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


class OracleHMMRepresentation(Representation):
    """Oracle view of a constrained HMM: sampling and feasibility on internal indices.

    Attributes
    ----------
    chmm : Oracle_CHMM
        The wrapped object.
    hmm : HMM_MatVecRepn
        The numeric HMM used for sampling and A*.
    constraints : list of OracleConstraint
        The constraints wrapped to take lists of internal hidden-state indices.
    """

    kind = "oracle"

    def __init__(self, cmodel):
        from conin.hidden_markov_model.chmm_oracle import Oracle_CHMM

        hmm = cmodel.model
        self.chmm = Oracle_CHMM(
            hmm=hmm.repn,
            constraints=list(cmodel.constraints),
            hidden_to_external=hmm.hidden_to_external,
            data=cmodel.data,
        )

    @property
    def hmm(self):
        return self.chmm.hmm

    @property
    def constraints(self):
        return self.chmm.constraints

    def generate_hidden(self, time_steps, max_failures=1000):
        """Sample a feasible sequence of internal hidden-state indices."""
        return self.chmm.generate_hidden(time_steps, max_failures=max_failures)

    def is_feasible(self, seq):
        """Check a full sequence of internal hidden-state indices."""
        return self.chmm.is_feasible(seq)

    def partial_is_feasible(self, *, T, seq):
        """Check whether a partial sequence of internal indices can still be completed."""
        return self.chmm.partial_is_feasible(T=T, seq=seq)

    def get_copy(self):
        """Return ``self``.

        This is safe because the object holds no per-query state: ``hmm`` is an
        ``HMM_MatVecRepn`` whose matrices are returned as copies, the wrapped
        constraints are closures over the user's constraints, and sampling draws
        from the global NumPy RNG (``HiddenMarkovModel.set_seed``), not from
        state stored here. Callers must not reassign attributes of ``chmm``.
        """
        return self


@register_representation("oracle", HiddenMarkovModel, families=(None, "oracle"))
def build_oracle_hmm_representation(cmodel):
    return OracleHMMRepresentation(cmodel)
