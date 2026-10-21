"""The ``"mvr"`` representation of a constrained HMM.

It wraps :class:`conin.hidden_markov_model.chmm_mvr.MVR_CHMM`, which aligns the
MVRs' hidden-state axes with the HMM.
"""

import copy

from conin.constraints.mvr import BaseMVR
from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


class MVRHMMRepresentation(Representation):
    """Built and aligned MVRs of a constrained HMM.

    Attributes
    ----------
    chmm : MVR_CHMM
        The wrapped object, for the torch MVR algorithms.
    mvrs : tuple of BaseMVR
        The built MVRs, aligned to the HMM's hidden-state order.
    hmm : HiddenMarkovModel
        The wrapped HMM.
    """

    kind = "mvr"

    def __init__(self, cmodel):
        from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

        hmm = cmodel.model
        data = cmodel.data
        # MVRConstraint factories are called with (hmm, data). Prebuilt MVRs
        # (also in the "mvr" family) are deep-copied: MVRs have mutable
        # state (e.g. ``mvr_timerange(..., inplace=True)`` sets
        # ``_time_range``), and the user's objects must not be changed through
        # the representation.
        built = [
            copy.deepcopy(c) if isinstance(c, BaseMVR) else c(hmm, data)
            for c in cmodel.constraints
        ]
        self.chmm = MVR_CHMM(hidden_markov_model=hmm, constraints=built, data=data)

    @property
    def mvrs(self):
        return tuple(self.chmm.constraints or ())

    @property
    def hmm(self):
        return self.chmm.hidden_markov_model


@register_representation("mvr", HiddenMarkovModel, families=(None, "mvr"))
def build_mvr_hmm_representation(cmodel):
    return MVRHMMRepresentation(cmodel)
