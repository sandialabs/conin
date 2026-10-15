import numpy as np

from conin.util import try_import
import conin.graphical_models

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import log_potential as log_potential_pgmpy


def log_potential(pgm, query_variables, evidence=None, **options):
    """
    A wrapper function that calls log_potential functions for specific libraries that
    conin interfaces with.
    """
    if isinstance(pgm, conin.graphical_models.DiscreteBayesianNetwork) or isinstance(
        pgm, conin.graphical_models.DiscreteMarkovNetwork
    ):
        return log_potential_conin(pgm, query_variables, evidence=evidence, **options)

    if pgmpy_available:
        if isinstance(pgm, pgmpy.models.DiscreteBayesianNetwork) or isinstance(
            pgm, pgmpy.models.DiscreteMarkovNetwork
        ):
            return log_potential_pgmpy(
                pgm, query_variables, evidence=evidence, **options
            )

    raise TypeError(f"Unexpected model type: {type(pgm)}")


def log_potential_conin(pgm, variables, evidence=None):
    """Log potential of a full assignment of a CONIN BN or MN."""

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
    ), "ERROR: cannot yet compute log_probability value with latent variables, which need to be marginalized."

    log_potential = 0.0
    if type(pgm) is conin.graphical_models.DiscreteBayesianNetwork:
        for cpd in pgm.cpds:
            if cpd.parents:
                if len(cpd.parents) == 1:
                    parent_index = data[cpd.parents[0]]
                else:
                    parent_index = tuple(data[node] for node in cpd.parents)
                x = cpd.values[parent_index][data[cpd.node]]
            else:
                x = cpd.values[data[cpd.node]]
            if x == 0.0:
                return -np.inf
            if x != 1.0:
                log_potential += np.log(x)

        return log_potential

    elif type(pgm) is conin.graphical_models.DiscreteMarkovNetwork:
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
