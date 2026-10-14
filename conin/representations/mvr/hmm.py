"""The ``"mvr"`` representation of a constrained HMM.

It wraps :class:`conin.hidden_markov_model.chmm_mvr.MVR_CHMM`, which aligns the
MVRs' hidden-state axes with the HMM. ``MVR_CHMM`` becomes this representation
in PR 4d.
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
        The wrapped object, for the torch MVR algorithms (unchanged until PR 4d).
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
        # MVRConstraint factories are called with (hmm, data); prebuilt MVRs
        # (also in the "mvr" family) are used as they are.
        built = [
            c if isinstance(c, BaseMVR) else c(hmm, data) for c in cmodel.constraints
        ]
        self.chmm = MVR_CHMM(hidden_markov_model=hmm, constraints=built, data=data)

    @property
    def mvrs(self):
        return tuple(self.chmm.constraints or ())

    @property
    def hmm(self):
        return self.chmm.hidden_markov_model

    def get_copy(self):
        """Return a copy with deep-copied MVRs that shares the (immutable) HMM.

        MVRs still have mutable state until PR 4d (e.g.
        ``mvr_timerange(..., inplace=True)`` sets ``_time_range``), so the
        cached MVRs are never handed out directly.
        """
        new = copy.copy(self)
        new.chmm = copy.copy(self.chmm)
        new.chmm.constraints = copy.deepcopy(self.chmm.constraints)
        return new


@register_representation("mvr", HiddenMarkovModel, families=(None, "mvr"))
def build_mvr_hmm_representation(cmodel):
    return MVRHMMRepresentation(cmodel)
