import munch
from conin.bayesian_network import create_BN_map_query_pyomo_model
from conin.dynamic_bayesian_network.dbn_to_bn import create_bn_from_dbn
from .model import ConstrainedDynamicDiscreteBayesianNetwork


def create_DDBN_map_query_pyomo_model(
    *, pgm, start=0, stop=1, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.pgm if isinstance(pgm, ConstrainedDynamicDiscreteBayesianNetwork) else pgm
    )

    bn = create_bn_from_dbn(dbn=pgm_, start=start, stop=stop)

    model = create_BN_map_query_pyomo_model(
        pgm=bn,
        variables=variables,
        evidence=evidence,
        var_index_map=bn._pyomo_index_names,
        **options,
    )

    if isinstance(pgm, ConstrainedDynamicDiscreteBayesianNetwork) and pgm.constraints:
        data = munch.Munch(
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            T=list(range(start, stop + 1)),
        )
        for func in pgm.constraints:
            model = func(model, data)

    return model
