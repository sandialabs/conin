from conin.markov_network import create_MN_map_query_model


def create_BN_map_query_model(
    *, pgm, variables=None, evidence=None, var_index_map=None
):
    MN = pgm.to_markov_model()
    model = create_MN_map_query_model(
        pgm=MN, variables=variables, evidence=evidence, var_index_map=var_index_map
    )
    return model
