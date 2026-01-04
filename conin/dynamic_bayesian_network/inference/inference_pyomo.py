import munch
from conin.bayesian_network.inference.inference_pyomo import (
    create_pyomo_map_query_model_BN,
)
from conin.dynamic_bayesian_network.dbn_to_bn import create_bn_from_dbn
from conin.dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork
import conin.markov_network


def create_pyomo_map_query_model_DDBN(
    *, pgm, start=0, stop=1, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.pgm if isinstance(pgm, ConstrainedDynamicDiscreteBayesianNetwork) else pgm
    )

    bn = create_bn_from_dbn(dbn=pgm_, start=start, stop=stop)

    model = create_pyomo_map_query_model_BN(
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


def inference_pyomo_map_query_DDBN(
    *,
    pgm,
    start=0,
    stop=1,
    variables=None,
    evidence=None,
    **options,
):
    model = create_pyomo_map_query_model_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )
    return conin.markov_network.inference.inference_pyomo.solve_pyomo_map_query_model(
        model, **options
    )
