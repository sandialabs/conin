import copy

from conin.exceptions import InvalidInputError
from . import chmm
from .mvr import BaseMVR, MVR_MatVecRepn


def _align_hidden_states(mvr, hidden_to_external):
    """Return an MVR with its numeric hidden axis aligned to the HMM."""
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


class MVR_CHMM(chmm.CHMM):
    """Constrained HMM whose MVR hidden axes are aligned when built."""

    def __init__(self, *, hidden_markov_model=None, constraints=None, data=None):
        if hidden_markov_model is None:
            raise InvalidInputError("hidden_markov_model is a required argument")
        if hidden_markov_model.repn is None:
            raise InvalidInputError(
                "hidden_markov_model.repn is missing "
                "Please run Load_model() with the correct start/trans/emit probs"
            )
        # Align constraints once so algorithms can index their arrays directly.
        if constraints:
            hidden_to_external = list(hidden_markov_model.hidden_to_external)
            hidden_states = set(hidden_to_external)
            aligned = []

            for i, mvr in enumerate(constraints):
                if not isinstance(mvr, BaseMVR):
                    raise InvalidInputError(f"Constraint {i} is not an MVR object")
                if set(mvr.hidden_states) != hidden_states:
                    raise InvalidInputError(
                        f"Hidden states of constraint {i} do not match those of hidden_markov_model"
                    )
                aligned.append(_align_hidden_states(mvr, hidden_to_external))

            constraints = aligned

        super().__init__(
            hidden_markov_model=hidden_markov_model, data=data, constraints=constraints
        )
