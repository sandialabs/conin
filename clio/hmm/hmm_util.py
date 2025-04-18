import numpy as np
from clio.hmm import HMM
from clio.util import Util


def random_hmm(hidden_states, observed_states, seed=None):
    if seed is not None:
        np.random.seed(seed)

    S = Util.normalize_dictionary({s: np.random.uniform() for s in hidden_states})

    T = Util.normalize_2d_dictionary(
        {(a, b): np.random.uniform() for a in hidden_states for b in hidden_states}
    )
    E = Util.normalize_2d_dictionary(
        {(s, o): np.random.uniform() for s in hidden_states for o in observed_states}
    )

    hmm = HMM()
    hmm.load_model(start_probs=S, transition_probs=T, emission_probs=E)
    return hmm
