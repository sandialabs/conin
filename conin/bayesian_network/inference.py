from conin.markov_network import create_MN_map_query_model


def create_BN_map_query_model(*, pgm, variables=None, evidence=None):
    MN = pgm.to_markov_model()
    model = create_MN_map_query_model(
        pgm=MN, variables=getattr(pgm, "_pyomo_node_index", None), evidence=evidence
    )
    return model
