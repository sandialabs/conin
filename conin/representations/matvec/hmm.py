"""The ``"matvec"`` representation of an HMM: its parameters as NumPy arrays."""

import numpy as np

from conin.graphical_models import HiddenMarkovModel
from conin.representations.base import Representation, register_representation
from conin.util.frozen import freeze


def _read_only(values):
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


class MatVecHMMRepresentation(Representation):
    """The parameters of an HMM as read-only arrays on internal indices.

    The index view of an HMM, and the only place that converts between
    state labels and indices: used by Viterbi, A*, sampling, the network-flow
    formulation and the torch MVR kernels.  Hidden state ``i`` is
    ``hidden_states[i]`` and observable state ``j`` is ``observable_states[j]``;
    :attr:`hidden_to_internal` and :attr:`observable_to_internal` map the labels
    back to indices.  The order is the model's (``model.hidden_states``).

    Parameters
    ----------
    model : HiddenMarkovModel
        The HMM.

    Attributes
    ----------
    model : HiddenMarkovModel
        The HMM.
    num_hidden_states, num_observable_states : int
        ``n`` and ``m``.
    hidden_states, observable_states : list
        Read-only lists of the hidden and observable labels by index.
    hidden_to_internal, observable_to_internal : dict
        Read-only maps from hidden and observable labels to their indices.
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
        # The model's label order is the index order.
        self.hidden_states = freeze(list(model.hidden_states))
        self.observable_states = freeze(list(model.observable_states))
        hidden, observed = self.hidden_states, self.observable_states
        self.num_hidden_states = len(hidden)
        self.num_observable_states = len(observed)
        self.hidden_to_internal = freeze({h: i for i, h in enumerate(hidden)})
        self.observable_to_internal = freeze({o: j for j, o in enumerate(observed)})
        start = model.start_probs
        transition = model.transition_probs
        emission = model.emission_probs
        self.start_vec = _read_only([start[h] for h in hidden])
        self.transition_mat = _read_only(
            [[transition[h1, h2] for h2 in hidden] for h1 in hidden]
        )
        self.emission_mat = _read_only(
            [[emission[h, o] for o in observed] for h in hidden]
        )
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
