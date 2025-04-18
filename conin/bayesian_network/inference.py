from conin.markov_network import pyomo_MN_inference_formulation


def pyomo_BN_map_query(*, pgm, variables=None, evidence=None):
    MN = pgm.to_markov_model()
    model = pyomo_MN_inference_formulation(
        pgm=MN, X=getattr(pgm, "_pyomo_node_index", None)
    )
    if evidence is not None:
        for k, v in evidence.items():
            model.X[k, v].fix(1)
    return model
