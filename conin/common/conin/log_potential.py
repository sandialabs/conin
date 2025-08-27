import numpy as np
import pandas as pd

import conin.markov_network
import conin.bayesian_network

def Log(x):
    if x == 0.0:
        return -np.inf
    return np.log(x)


def log_potential(pgm, variables, evidence=None):

    data = {k: [v] for k, v in variables.items()}
    if evidence:
        data.update({k: [v] for k, v in evidence.items()})

    assert len(pgm.nodes) <= len(data), f"Cannot specify more node values than are in the pgmpy model: num_nodes={len(pgm.nodes)} num_values={len(data)}"
    assert set(data.keys()).issubset(pgm.nodes), "Expecting that data keys to be a subset of the PGM nodes"

    # TODO - remove this error
    assert len(pgm.nodes) == len(data), f"ERROR: cannot yet compute log_probability value with latent variables, which need to be marginalized."

    if type(pgm) is conin.bayesian_network.DiscreteBayesianNetwork:
        assert len(pgm.nodes) == len(data), f"ERROR: cannot yet compute log_potential value with latent variables, which need to be marginalized."

        df = pd.DataFrame.from_dict(data)

        return None
        #return pgmpy.metrics.log_likelihood_score(pgm, df)

    elif type(pgm) is conin.markov_network.DiscreteMarkovNetwork:
        return None

        log_potential = 0.0
        for factor in pgm.factors:
            scope_vars = factor.nodes
            values = [(var, val[0]) for var, val in data.items() if var in scope_vars]
            reduced_factor = factor.reduce(values, inplace=False)
            assert (
                len(reduced_factor.variables) == 0
            ), f"Expecting evidence for all variables in factor {factor}"

            x = reduced_factor.values
            if x == 0.0:
                return -np.inf
            if x != 1.0:
                log_potential += np.log(x)
        return log_potential

    else:
        raise RuntimeError(f"Unexpected model type: {type(pgm)}")
