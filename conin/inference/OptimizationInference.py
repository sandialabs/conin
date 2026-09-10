from ovld import ovld
from conin.util import try_import
from conin.hidden_markov_model import (
    HiddenMarkovModel,
    ConstrainedHiddenMarkovModel,
    CHMM,
)

from conin.markov_network import (
    DiscreteMarkovNetwork,
    ConstrainedDiscreteMarkovNetwork,
)
from .mn import (
    inference_pyomo_map_query_MN,
)
from conin.bayesian_network import (
    DiscreteBayesianNetwork,
    ConstrainedDiscreteBayesianNetwork,
)
from .bn import (
    inference_pyomo_map_query_BN,
)
from conin.dynamic_bayesian_network import (
    DynamicDiscreteBayesianNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
)
from .dbn import (
    inference_pyomo_map_query_DDBN,
)

from .hmm import (
    inference_pyomo_map_query_HMM,
)

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin


@ovld
def _map_query_IntegerProgram(
    pgm,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    """Compute a MAP assignment using Pyomo-based optimization with multiple dispatch."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: DiscreteMarkovNetwork, ConstrainedDiscreteMarkovNetwork, "
        f"DiscreteBayesianNetwork, ConstrainedDiscreteBayesianNetwork, "
        f"DynamicDiscreteBayesianNetwork, ConstrainedDynamicDiscreteBayesianNetwork, "
        f"HiddenMarkovModel, ConstrainedHiddenMarkovModel, CHMM."
    )


# Static models: DiscreteMarkovNetwork and ConstrainedDiscreteMarkovNetwork


@ovld
def _map_query_IntegerProgram(
    pgm: DiscreteMarkovNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_IntegerProgram(
    pgm: ConstrainedDiscreteMarkovNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# Static models: DiscreteBayesianNetwork and ConstrainedDiscreteBayesianNetwork


@ovld
def _map_query_IntegerProgram(
    pgm: DiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_IntegerProgram(
    pgm: ConstrainedDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# Dynamic models: DynamicDiscreteBayesianNetwork and ConstrainedDynamicDiscreteBayesianNetwork


@ovld
def _map_query_IntegerProgram(
    pgm: DynamicDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )


@ovld
def _map_query_IntegerProgram(
    pgm: ConstrainedDynamicDiscreteBayesianNetwork,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )


# Dynamic models: HiddenMarkovModel, ConstrainedHiddenMarkovModel, and CHMM


@ovld
def _map_query_IntegerProgram(
    pgm: HiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )


@ovld
def _map_query_IntegerProgram(
    pgm: ConstrainedHiddenMarkovModel,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )


@ovld
def _map_query_IntegerProgram(
    pgm: CHMM,
    *,
    variables=None,
    evidence=None,
    show_progress=False,
    timing=False,
    start=0,
    stop=None,
    options={},
):
    return inference_pyomo_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        **options,
    )


# pgmpy model conversions (if pgmpy is available)

if pgmpy_available:

    @ovld
    def _map_query_IntegerProgram(
        pgm: pgmpy.models.DiscreteMarkovNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        converted = convert_pgmpy_to_conin(pgm)
        return _map_query_IntegerProgram(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            options=options,
        )

    @ovld
    def _map_query_IntegerProgram(
        pgm: pgmpy.models.DiscreteBayesianNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        converted = convert_pgmpy_to_conin(pgm)
        return _map_query_IntegerProgram(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            options=options,
        )

    @ovld
    def _map_query_IntegerProgram(
        pgm: pgmpy.models.DynamicBayesianNetwork,
        *,
        variables=None,
        evidence=None,
        show_progress=False,
        timing=False,
        start=0,
        stop=None,
        options={},
    ):
        converted = convert_pgmpy_to_conin(pgm)
        return _map_query_IntegerProgram(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            start=start,
            stop=stop,
            options=options,
        )
