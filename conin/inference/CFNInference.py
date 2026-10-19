from ovld import ovld
from conin.util import try_import

from conin.graphical_models import HiddenMarkovModel
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models import DiscreteMarkovNetwork
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from .toulbar2 import (
    inference_toulbar2_map_query_MN,
    inference_toulbar2_map_query_BN,
    inference_toulbar2_map_query_DDBN,
    inference_toulbar2_map_query_HMM,
)
from conin.graphical_models import DiscreteBayesianNetwork
from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.graphical_models import DynamicDiscreteBayesianNetwork
from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin


@ovld
def _map_query_Toulbar2(
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
    """Compute a MAP assignment using Toulbar2-based cost function network inference."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: DiscreteMarkovNetwork, ConstrainedDiscreteMarkovNetwork, "
        f"DiscreteBayesianNetwork, ConstrainedDiscreteBayesianNetwork, "
        f"DynamicDiscreteBayesianNetwork, ConstrainedDynamicDiscreteBayesianNetwork, "
        f"HiddenMarkovModel, ConstrainedHiddenMarkovModel."
    )


# Static models: DiscreteMarkovNetwork and ConstrainedDiscreteMarkovNetwork


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# Static models: DiscreteBayesianNetwork and ConstrainedDiscreteBayesianNetwork


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# Dynamic models: DynamicDiscreteBayesianNetwork and ConstrainedDynamicDiscreteBayesianNetwork


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# Dynamic models: HiddenMarkovModel and ConstrainedHiddenMarkovModel


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


@ovld
def _map_query_Toulbar2(
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
    return inference_toulbar2_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        timing=timing,
        **options,
    )


# pgmpy model conversions (if pgmpy is available)

if pgmpy_available:

    @ovld
    def _map_query_Toulbar2(
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
        return _map_query_Toulbar2(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            start=start,
            stop=stop,
            options=options,
        )

    @ovld
    def _map_query_Toulbar2(
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
        return _map_query_Toulbar2(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            start=start,
            stop=stop,
            options=options,
        )

    @ovld
    def _map_query_Toulbar2(
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
        return _map_query_Toulbar2(
            converted,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            start=start,
            stop=stop,
            options=options,
        )
