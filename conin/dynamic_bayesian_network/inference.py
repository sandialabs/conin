from conin.bayesian_network import create_BN_map_query_model
from conin.dynamic_bayesian_network.dbn_to_bn import create_bn_from_dbn


def create_DBN_map_query_model(
    *, pgm, start=0, stop=1, variables=None, evidence=None, **options
):
    bn = create_bn_from_dbn(dbn=pgm, start=start, stop=stop)

    return create_BN_map_query_model(
        pgm=bn,
        variables=variables,
        evidence=evidence,
        var_index_map=bn._pyomo_index_names,
        **options,
    )
