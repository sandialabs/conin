"""The ``"mvr"`` representation of a constrained HMM.

It builds the MVRs of the constrained model and aligns their hidden-state
axes with the HMM, for the torch MVR algorithms
(:mod:`conin.inference.mvr`, :mod:`conin.learning.mvr`,
:mod:`conin.sampling.mvr` and :mod:`conin.other_queries.mvr`).
"""

import copy

from conin.constraints.mvr import BaseMVR, MVR_MatVecRepn
from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


def align_hidden_states(mvr, hidden_to_external):
    """Return an MVR whose numeric hidden axis follows ``hidden_to_external``.

    Parameters
    ----------
    mvr : HomMVR or InhomMVR
        An MVR over the same hidden states as the HMM.
    hidden_to_external : list
        The HMM's hidden-state labels in internal-index order.

    Returns
    -------
    HomMVR or InhomMVR
        ``mvr`` itself if its hidden states are already in that order,
        otherwise a shallow copy with a permuted array representation.
    """
    if list(mvr.hidden_states) == hidden_to_external:
        return mvr

    position = {h: i for i, h in enumerate(mvr.hidden_states)}
    perm = [position[h] for h in hidden_to_external]

    repn = mvr.repn
    upd_array = repn.upd_array

    aligned = copy.copy(mvr)
    aligned.hidden_states = list(hidden_to_external)

    # Validation is skipped, since the existing MVR should have validated already.
    aligned._repn = MVR_MatVecRepn(
        ini_array=repn.ini_array[perm],
        upd_array=(
            [upd_t[perm] for upd_t in upd_array]
            if isinstance(upd_array, list)
            else upd_array[perm]
        ),
        evl_array=repn.evl_array,
        check_errors=False,
    )

    return aligned


class MVRHMMRepresentation(Representation):
    """The built MVRs of a constrained HMM, aligned to the HMM.

    ``MVRConstraint`` factories are called with ``(hmm, cmodel.data)``.
    Prebuilt MVRs are deep-copied, so the user's MVR objects are never
    changed through the representation.  Each MVR's hidden axis is then put
    in the HMM's internal hidden-state order, so the torch algorithms can
    index the MVR arrays and the HMM arrays the same way.

    Attributes
    ----------
    model : HiddenMarkovModel
        The wrapped HMM.
    mvrs : tuple of BaseMVR
        The built MVRs, aligned to the HMM's hidden-state order, in the order
        of the constrained model's constraints (empty without constraints).

    Notes
    -----
    The representation holds no per-query data: the time horizon,
    observations, dtype and device are arguments of the algorithms, which
    build their torch tensors from ``mvrs`` on each call.  Nothing here
    changes after the build, so one representation can serve several
    queries.

    Raises
    ------
    InvalidInputError
        If a constraint does not give an MVR, or an MVR's hidden states are
        not those of the HMM.
    """

    kind = "mvr"

    def __init__(self, cmodel):
        hmm = cmodel.model
        hidden_to_external = list(hmm.hidden_to_external)
        hidden_states = set(hidden_to_external)

        mvrs = []
        for i, c in enumerate(cmodel.constraints):
            if isinstance(c, BaseMVR):
                mvr = copy.deepcopy(c)
            elif callable(c):
                mvr = c(hmm, cmodel.data)
            else:
                mvr = c
            if not isinstance(mvr, BaseMVR):
                raise InvalidInputError(f"Constraint {i} is not an MVR object")
            if set(mvr.hidden_states) != hidden_states:
                raise InvalidInputError(
                    f"Hidden states of constraint {i} do not match those of the "
                    "hidden Markov model"
                )
            mvrs.append(align_hidden_states(mvr, hidden_to_external))

        self.model = hmm
        self.mvrs = tuple(mvrs)


@register_representation("mvr", HiddenMarkovModel, families=(None, "mvr"))
def build_mvr_hmm_representation(cmodel):
    return MVRHMMRepresentation(cmodel)
