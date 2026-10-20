import copy

from ovld import ovld
from pyomo.common.timing import TicTocTimer

from conin.util import try_import

from conin.graphical_models import HiddenMarkovModel
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.graphical_models import DiscreteMarkovNetwork
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.graphical_models import DiscreteBayesianNetwork
from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.graphical_models import DynamicDiscreteBayesianNetwork
from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from .variable_elimination import (
    inference_variable_elimination_map_query_MN,
    inference_variable_elimination_map_query_BN,
    inference_variable_elimination_map_query_DDBN,
    inference_variable_elimination_map_query_HMM,
    solve_variable_elimination_map_query_model,
)
from .variable_elimination.solve import _with_timing

with try_import() as pgmpy_available:
    import pgmpy.models
    from conin.common.pgmpy import convert_pgmpy_to_conin


def _execute_pgmpy_model_query(
    pgm,
    variables,
    evidence,
    show_progress,
    timing,
):
    """Execute MAP query for native pgmpy models (no conversion needed)."""

    def _execute():
        evidence_prepared = copy.copy(evidence) if evidence else {}
        if variables is None:
            if evidence_prepared:
                vars_determined = [
                    node for node in pgm.nodes if node not in evidence_prepared
                ]
            else:
                vars_determined = pgm.nodes
        else:
            vars_determined = variables

        if timing:
            TicTocTimer().tic("Created PGMPY model")

        return solve_variable_elimination_map_query_model(
            pgm,
            variables=vars_determined,
            evidence=evidence_prepared,
            show_progress=show_progress,
        )

    return _with_timing(timing, _execute)


# -----------------------------------------------------------------------------------------
# map_query functions
# -----------------------------------------------------------------------------------------


@ovld
def _map_query_VariableElimination(
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
    """Compute a MAP assignment using pgmpy's Variable Elimination with multiple dispatch."""
    raise TypeError(
        f"Unsupported model type: {type(pgm)}. "
        f"Expected one of: DiscreteMarkovNetwork, ConstrainedDiscreteMarkovNetwork, "
        f"DiscreteBayesianNetwork, ConstrainedDiscreteBayesianNetwork, "
        f"DynamicDiscreteBayesianNetwork, ConstrainedDynamicDiscreteBayesianNetwork, "
        f"HiddenMarkovModel, ConstrainedHiddenMarkovModel, "
        f"pgmpy.models.DiscreteMarkovNetwork, pgmpy.models.DiscreteBayesianNetwork, "
        f"pgmpy.models.DynamicBayesianNetwork."
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_MN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_BN(
        pgm=pgm,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    return inference_variable_elimination_map_query_DDBN(
        pgm=pgm,
        start=start,
        stop=stop,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    # The HMM path unrolls over len(evidence) and has always ignored ``stop``.
    return inference_variable_elimination_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=None,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


@ovld
def _map_query_VariableElimination(
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
    # The HMM path unrolls over len(evidence) and has always ignored ``stop``.
    return inference_variable_elimination_map_query_HMM(
        pgm=pgm,
        start=start,
        stop=None,
        variables=variables,
        evidence=evidence,
        show_progress=show_progress,
        timing=timing,
        **options,
    )


if pgmpy_available:

    @ovld
    def _map_query_VariableElimination(
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
        return _execute_pgmpy_model_query(
            pgm,
            variables,
            evidence,
            show_progress,
            timing,
        )

    @ovld
    def _map_query_VariableElimination(
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
        return _execute_pgmpy_model_query(
            pgm,
            variables,
            evidence,
            show_progress,
            timing,
        )

    @ovld
    def _map_query_VariableElimination(
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
        return inference_variable_elimination_map_query_DDBN(
            pgm=convert_pgmpy_to_conin(pgm),
            start=start,
            stop=stop,
            variables=variables,
            evidence=evidence,
            show_progress=show_progress,
            timing=timing,
            **options,
        )
