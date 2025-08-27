import numpy as np
import pandas as pd

try:
    from pgmpy import config
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork, MarkovNetwork  # DiscreteMarkovNetwork
    import pgmpy.metrics
    pgmpy_available=True
except:
    pgmpy_available=False

from .custom_VE import logprob_VE



def set_backend_(torch_bool):
    if torch_bool:
        config.set_backend("torch")
    else:
        config.set_backend("numpy")


def Log(x):
    if x == 0.0:
        return -np.inf
    return np.log(x)


def log_potential(pgm, variables, evidence=None):
    data = {k: [v] for k, v in variables.items()}
    if evidence:
        data.update({k: [v] for k, v in evidence.items()})

    if hasattr(pgm, "pgmpy"):
        # This is a hack.  If we load a model with pyagrum, then we re-load it with pgmpy.
        pgm = pgm.pgmpy

    assert len(pgm.nodes()) <= len(
        data
    ), f"Cannot specify more node values than are in the pgmpy model: num_nodes={len(pgm.nodes())} num_values={len(data)}"
    assert set(data.keys()).issubset(
        pgm.nodes()
    ), "Expecting that data keys to be a subset of the PGM nodes"

    # TODO - remove this error
    assert len(pgm.nodes()) == len(
        data
    ), f"ERROR: cannot yet compute log_probability value with latent variables, which need to be marginalized."

    if type(pgm) is pgmpy.models.DiscreteBayesianNetwork:
        assert len(pgm.nodes()) == len(
            data
        ), f"ERROR: cannot yet compute log_potential value with latent variables, which need to be marginalized."

        df = pd.DataFrame.from_dict(data)

        # WEH - I don't know why I'm getting an error when using pytorch
        config.set_backend("numpy")
        return pgmpy.metrics.log_likelihood_score(pgm, df)

    elif type(pgm) is pgmpy.models.MarkovNetwork:
        log_potential = 0.0
        for factor in pgm.get_factors():
            scope_vars = factor.scope()
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


def log_probability(pgm, query_vars, query_values, evidence=None):
    """
    For a subset of variables X, a list of assignement [X=x1, X=x2,...], and evidence E=e, computes:
        P(X=x_i|E=e)

    In pgmpy, the output of VariableElimination will return a single giant CPD over the query variables.
    We compute this once, and then plug each each assignment x_i.

    This computation of the giant CPD is still very memory inefficient and will blow up.
    Working on writing a custom version  of VE that instead just saves the reduced factors.

    IN:
    query_vars: list of strings. List of names of queried variables.
    query_values: list of lists. each sublist is a value assignment to the variables in query_vars.
    evidence: dict. var:value
    pgm: should be a pgmpy DiscreteBayesianNetwork.

    OUT:
    ret_val: list of log probs for each assignment in query_vals
    """

    if not isinstance(pgm, DiscreteBayesianNetwork):
        raise TypeError("pgm must be pgmpy DiscreteBayesianNetwork")

    infer_pgm = VariableElimination(pgm)
    final_factor = infer_pgm.query(variables=query_vars, evidence=evidence)

    ret_val = []
    for values in query_values:
        query_dict = dict(zip(query_vars, values))
        ret_val.append(Log(final_factor.get_value(**query_dict)))

    return ret_val


def log_prob_v2(
    pgm, query_vars, query_values, evidence=None, elimination_order="MinFill"
):
    """
    Same as log_prob function. If pgm is DiscreteBN, converts it first to a DiscreteMN.
    """

    if isinstance(pgm, DiscreteBayesianNetwork):
        pgm = pgm.to_markov_model()

    if not isinstance(pgm, MarkovNetwork):  # DiscreteMarkovNetwork
        raise TypeError(
            "pgm must be pgmpy DiscreteBayesianNetwork or DiscreteMarkovNetwork"
        )

    infer_pgm = logprob_VE(pgm)

    logprob_list = infer_pgm.logprobs(
        variables=query_vars,
        variable_values=query_values,
        evidence=evidence,
        elimination_order=elimination_order,
    )
    return logprob_list
