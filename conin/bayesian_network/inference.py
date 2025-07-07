try:
    import pgmpy.models
    import pgmpy.inference
except:
    pass
from conin.markov_network import create_MN_map_query_model


def create_BN_map_query_model(
    *,
    pgm,
    variables=None,
    evidence=None,
    var_index_map=None,
    **options,
):
    prune_network = options.pop("prune_network", True)
    create_MN = options.pop("create_MN", False)

    if prune_network:
        """
        Apply the pruning methods described here:

        M. Baker, T. E. Boult. 1990. Pruning bayesian networks for efficient computation. 
        In Proc. of the Sixth Annual Conf. on Uncertainty in Artificial Intelligence (UAI '90). 
        Elsevier Science Inc., USA, 225–232.
        https://arxiv.org/abs/1304.1112
        """
        inf = pgmpy.inference.Inference(pgm)
        pgm, evidence = inf._prune_bayesian_model(
            [] if variables is None else variables, evidence
        )

    if variables or evidence or create_MN:
        #
        # We need to create a Markov network if (1) the user asks us to, or (2) we have 'variables' or
        # 'evidence' specified.  In (2), we need to reduce the factors and apply variable elimination to
        # eliminate unspecified variables.
        #
        MN = pgm.to_markov_model()
    else:
        #
        # By default, we avoid creating a complete Markov model.  Rather, we
        # create the skeleton of a model that is used to setup the integer program.
        #
        MN = pgmpy.models.MarkovNetwork()
        MN.add_nodes_from(pgm.nodes())
        MN.add_factors(*[cpd.to_factor() for cpd in pgm.cpds])

    model = create_MN_map_query_model(
        pgm=MN,
        variables=variables,
        evidence=evidence,
        var_index_map=var_index_map,
    )
    return model
