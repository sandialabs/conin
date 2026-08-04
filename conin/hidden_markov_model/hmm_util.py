import numpy as np
from conin.hidden_markov_model import HiddenMarkovModel
from conin.util import Util


def random_hmm(*, hidden_states, observed_states, seed=None):
    """Generate a random hidden Markov model.

    Parameters
    ----------
    hidden_states : iterable
        Hidden state labels for the generated model.
    observed_states : iterable
        Observable state labels for the generated model.
    seed : int, optional
        Random seed used for reproducible sampling.

    Returns
    -------
    HiddenMarkovModel
        Hidden Markov model with randomly generated start, transition, and
        emission probabilities.
    """
    if seed is not None:
        np.random.seed(seed)

    S = Util.normalize_dictionary({s: np.random.uniform() for s in hidden_states})

    T = Util.normalize_2d_dictionary(
        {(a, b): np.random.uniform() for a in hidden_states for b in hidden_states}
    )
    E = Util.normalize_2d_dictionary(
        {(s, o): np.random.uniform() for s in hidden_states for o in observed_states}
    )

    hmm = HiddenMarkovModel()
    hmm.load_model(start_probs=S, transition_probs=T, emission_probs=E)
    return hmm
