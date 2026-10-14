import munch
import conin.markov_network
from conin.graphical_models import create_bn_from_dbn
from conin.dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork

from conin.inference.bn.inference_pyomo import create_pyomo_map_query_model_BN


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
        conin.inference.mn.inference_pyomo.add_constraints(
            pgm=bn, constraints=pgm.constraints, model=model, data=data
        )

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
    use_cut_generation = options.pop("use_cut_generation", False)
    model = create_pyomo_map_query_model_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        omit_c5=use_cut_generation,
        **options,
    )
    return conin.inference.mn.inference_pyomo.solve_pyomo_map_query_model(
        model, evidence=evidence, **options
    )
