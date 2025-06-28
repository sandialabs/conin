try:
    import pgmpy.models
except:
    pass
from conin.markov_network import create_MN_map_query_model


def create_BN_map_query_model(
    *, pgm, variables=None, evidence=None, var_index_map=None, create_MN=False
):
    
    if create_MN:
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
