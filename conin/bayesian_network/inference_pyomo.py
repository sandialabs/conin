import warnings
import munch
from pyomo.common.timing import TicTocTimer

import conin.markov_network
from .model import ConstrainedDiscreteBayesianNetwork


def create_pyomo_map_query_model_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    var_index_map=None,
    timing=False,
    **options,
):
    """Create a MAP query model.

    Parameters
    ----------
    variables : list, optional
        Nodes for which to compute the MAP estimate.
    evidence : dict, optional
        Observed node assignments.
    timing : bool, optional
        Whether to collect timing information.
    var_index_map : dict, optional
        Dictionary mapping variable indices to Pyomo variable data objects
    **options : dict, optional
        Additional keyword arguments forwarded to
        :func:`create_BN_map_query_model`.

    Returns
    -------
    conin.bayesian_network.inference.BNMapQueryModel
        Constrained MAP query model.
    """
    if timing:
        timer = TicTocTimer()
        timer.tic("create_pyomo_map_query_model_BN - START")
    prune_network = options.pop("prune_network", False)
    create_MN = options.pop("create_MN", False)

    pgm_ = pgm.pgm if isinstance(pgm, ConstrainedDiscreteBayesianNetwork) else pgm

    if prune_network:
        """
        Apply the pruning methods described here:

        M. Baker, T. E. Boult. 1990. Pruning bayesian networks for efficient computation.
        In Proc. of the Sixth Annual Conf. on Uncertainty in Artificial Intelligence (UAI '90).
        Elsevier Science Inc., USA, 225–232.
        https://arxiv.org/abs/1304.1112

        Note: prune_network defaults to False because the implementation of this method in pgmpy is
        slow for large models.
        """
        raise RuntimeError("Pruning is not currently supported with CONIN models.")

        # inf = pgmpy.inference.Inference(pgm)
        # pgm, evidence = inf._prune_bayesian_model(
        #    [] if variables is None else variables, evidence
        # )
        # if timing:
        #    timer.toc("Created pruned model")

    if variables and len(variables) == len(pgm_.nodes):
        assert set(variables) == set(
            pgm_.nodes
        ), "Mismatch in the specified variables and the nodes in the model"
        # We continue with 'variables' set to None, which is a special case recognized below
        variables = None

    if variables or evidence or create_MN:
        #
        # We need to create a Markov network if (1) the user asks us to, or (2) we have 'variables' or
        # 'evidence' specified.  In (2), we need to reduce the factors and apply variable elimination to
        # eliminate unspecified variables.
        #
        raise RuntimeError(
            "CONIN does not currently support the generation of discrete Markov networks using variable elimination to prune unspecified variables."
        )
        MN = pgm_.to_markov_model()
        if timing:
            timer.toc("Created Markov network from Bayesian network")
    else:
        #
        # By default, we avoid creating a complete Markov model.  Rather, we
        # create the skeleton of a model that is used to setup the integer program.
        #
        MN = conin.markov_network.DiscreteMarkovNetwork()
        MN.states = pgm_.states
        MN.factors = [cpd.to_factor() for cpd in pgm_.cpds]
        if timing:
            timer.toc("Created skeleton Markov network")

    model = conin.markov_network.inference_pyomo.create_pyomo_map_query_model_MN(
        pgm=MN,
        variables=variables,
        evidence=evidence,
        var_index_map=var_index_map,
        timing=timing,
    )

    if isinstance(pgm, ConstrainedDiscreteBayesianNetwork) and pgm.constraints:
        data = munch.Munch(variables=variables, evidence=evidence)
        for func in pgm.constraints:
            model = func(model, data)

    if timing:
        timer.toc("create_pyomo_map_query_model_BN - STOP")
    return model
