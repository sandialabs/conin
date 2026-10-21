"""The ``"matvec"`` representation of an HMM: its parameters as NumPy arrays."""

import numpy as np

from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation


def _read_only(values):
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


class MatVecHMMRepresentation(Representation):
    """The parameters of an HMM as read-only arrays on internal indices.

    Used by Viterbi, A*, sampling and the torch MVR kernels.  Hidden state
    ``i`` is ``model.hidden_to_external[i]`` and observed state ``j`` is
    ``model.observed_to_external[j]``.

    Parameters
    ----------
    model : HiddenMarkovModel
        The HMM.

    Attributes
    ----------
    model : HiddenMarkovModel
        The HMM.
    num_hidden_states, num_observed_states : int
        ``n`` and ``m``.
    start_vec : numpy.ndarray
        Starting probabilities, shape ``(n,)``.
    transition_mat : numpy.ndarray
        Transition probabilities, shape ``(n, n)``.
    emission_mat : numpy.ndarray
        Emission probabilities, shape ``(n, m)``.
    log_start_vec, log_transition_mat, log_emission_mat : numpy.ndarray
        Their natural logarithms (``-inf`` where a probability is zero).

    Notes
    -----
    All arrays are ``float64`` and read-only (also after ``copy.deepcopy``
    or pickling), so one representation can serve several queries.
    """

    kind = "matvec"

    def __init__(self, model):
        self.model = model
        self.num_hidden_states = model.num_hidden_states
        self.num_observed_states = model.num_observed_states
        self.start_vec = _read_only(model.start_vec)
        self.transition_mat = _read_only(model.transition_mat)
        self.emission_mat = _read_only(model.emission_mat)
        with np.errstate(divide="ignore"):
            self.log_start_vec = _read_only(np.log(self.start_vec))
            self.log_transition_mat = _read_only(np.log(self.transition_mat))
            self.log_emission_mat = _read_only(np.log(self.emission_mat))

    _ARRAYS = (
        "start_vec",
        "transition_mat",
        "emission_mat",
        "log_start_vec",
        "log_transition_mat",
        "log_emission_mat",
    )

    def __setstate__(self, state):
        # copy.deepcopy and pickle give writeable arrays; make them read-only again.
        self.__dict__.update(state)
        for name in self._ARRAYS:
            getattr(self, name).setflags(write=False)


@register_representation("matvec", HiddenMarkovModel)
def build_matvec_hmm_representation(cmodel):
    return MatVecHMMRepresentation(cmodel.model)
