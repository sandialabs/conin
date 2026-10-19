import munch
import conin.constrained_models
from conin.graphical_models import create_bn_from_dbn
from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from .bn import (
    create_toulbar2_map_query_model_BN,
)
from .solve import add_constraints, solve_toulbar2_map_query_model


def create_toulbar2_map_query_model_DDBN(
    *, pgm, start=0, stop=1, variables=None, evidence=None, **options
):
    pgm_ = (
        pgm.model if isinstance(pgm, ConstrainedDynamicDiscreteBayesianNetwork) else pgm
    )

    bn = create_bn_from_dbn(dbn=pgm_, start=start, stop=stop)

    model = create_toulbar2_map_query_model_BN(
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
        add_constraints(
            pgm=bn, constraints=pgm.constraints, model=model, data=data
        )

    return model


def inference_toulbar2_map_query_DDBN(
    *,
    pgm,
    start=0,
    stop=1,
    variables=None,
    evidence=None,
    write_uai_file=None,
    **options,
):
    model = create_toulbar2_map_query_model_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        write_uai_file=write_uai_file,
        **options,
    )
    return solve_toulbar2_map_query_model(model, **options)
