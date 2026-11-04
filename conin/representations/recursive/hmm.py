"""The ``"recursive"`` representation of a constrained HMM (used by recursive A*)."""

from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


class RecursiveHMMRepresentation(Representation):
    """The recursive constraints of a constrained HMM, combined into one.

    The combined state is the tuple of the per-constraint states (``()``
    without constraints), and a state is (partially) feasible iff every
    constraint's state is.

    Attributes
    ----------
    model : HiddenMarkovModel
        The wrapped HMM.
    constraints : tuple of RecursiveConstraint
        The constraints of the constrained model.

    Notes
    -----
    The representation holds no per-query data, so one representation can
    serve several searches.
    """

    kind = "recursive"

    def __init__(self, cmodel):
        self.model = cmodel.model
        self.constraints = cmodel.constraints

    def initial(self, h):
        """The combined state after the first hidden state ``h`` (an external label)."""
        return tuple(c.initial(h) for c in self.constraints)

    def update(self, state, h):
        """The combined state after appending the hidden state ``h``."""
        return tuple(c.update(s, h) for c, s in zip(self.constraints, state))

    def feasible(self, state):
        """Whether a complete sequence with this combined state is feasible."""
        return all(c.feasible(s) for c, s in zip(self.constraints, state))

    def feasible_partial(self, state, t, T):
        """Whether a prefix of length ``t`` may still be completed to length ``T``."""
        return all(c.feasible_partial(s, t, T) for c, s in zip(self.constraints, state))


@register_representation("recursive", HiddenMarkovModel, families=(None, "recursive"))
def build_recursive_hmm_representation(cmodel):
    return RecursiveHMMRepresentation(cmodel)
