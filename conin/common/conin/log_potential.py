import numpy as np

# import pandas as pd

import conin.markov_network
import conin.bayesian_network


def log_potential(pgm, variables, evidence=None):

    data = {k: v for k, v in variables.items()}
    if evidence:
        data.update({k: v for k, v in evidence.items()})

    assert len(pgm.nodes) <= len(
        data
    ), f"Cannot specify more node values than are in the pgmpy model: num_nodes={len(pgm.nodes)} num_values={len(data)}"
    assert set(data.keys()).issubset(
        pgm.nodes
    ), "Expecting that data keys to be a subset of the PGM nodes"

    # TODO - remove this error
    assert len(pgm.nodes) == len(
        data
    ), f"ERROR: cannot yet compute log_probability value with latent variables, which need to be marginalized."

    log_potential = 0.0
    if type(pgm) is conin.bayesian_network.DiscreteBayesianNetwork:
        for cpd in pgm.cpds:
            if cpd.evidence:
                x = cpd.values[tuple(data[node] for node in cpd.evidence)][
                    data[cpd.variable]
                ]
            else:
                x = cpd.values[data[cpd.variable]]
            if x == 0.0:
                return -np.inf
            if x != 1.0:
                log_potential += np.log(x)

        return log_potential

    elif type(pgm) is conin.markov_network.DiscreteMarkovNetwork:
        for factor in pgm.factors:
            if len(factor.nodes) == 1:
                x = factor.values[data[factor.nodes[0]]]
            else:
                x = factor.values[tuple(data[node] for node in factor.nodes)]
            if x == 0.0:
                return -np.inf
            if x != 1.0:
                log_potential += np.log(x)
        return log_potential

    else:
        raise RuntimeError(f"Unexpected model type: {type(pgm)}")
