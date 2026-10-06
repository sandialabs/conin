from ovld import ovld
from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
from conin.sampling.dbn.sample import _sample as _sample_dbn


@ovld
def _sample(G, *, N=10, T=10, return_indices=False):
    """
    Sample N traces of length T from the pgm G
    """
    raise TypeError(
        f"Unsupported model type: {type(G)}. "
        f"Expected one of: DynamicDiscreteBayesianNetwork"
    )


@ovld
def _sample(G: DynamicDiscreteBayesianNetwork, *, N=10, T=10, return_indices=False):
    return _sample_dbn(G, N=N, T=T, return_indices=return_indices)
