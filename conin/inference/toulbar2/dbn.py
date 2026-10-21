"""toulbar2 MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from conin.inference.query_checks import full_query_variables, reject_pruning_options

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
)


def _map_query_toulbar2_dbn(repn, *, variables, evidence, write_uai_file, options):
    """Solve a MAP query on the ``"toulbar2"`` representation of a DBN."""
    if not pytoulbar2_available:
        return _not_available()
    reject_pruning_options(options)
    variables = full_query_variables(variables, repn.nodes)
    model = _create_model(
        repn,
        evidence=evidence,
        variables=variables,
        timing=options.get("timing", False),
        write_uai_file=write_uai_file,
        options=options,
    )
    return solve_toulbar2_map_query_model(model, **options)


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
    """MAP query on a DBN unrolled over ``start..stop`` with toulbar2.

    Parameters
    ----------
    pgm : DynamicDiscreteBayesianNetwork or ConstrainedDynamicDiscreteBayesianNetwork
    start, stop : int, optional
        The time steps to unroll (structure params of the ``"toulbar2"`` representation).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported (``RuntimeError``).
    evidence : dict, optional
        ``{(node, t): state}``.
    write_uai_file : str, optional
        Write the UAI file that toulbar2 reads to this path.
    **options
        ``timing``, ``verbose``, ``prune_network`` and ``create_MN`` (as for
        :func:`~conin.inference.toulbar2.inference_toulbar2_map_query_BN`) and
        the options of
        :func:`~conin.inference.toulbar2.solve_toulbar2_map_query_model`.
    """
    if not pytoulbar2_available:
        return _not_available()

    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    return _map_query_toulbar2_dbn(
        cmodel.representation("toulbar2", start=start, stop=stop),
        variables=variables,
        evidence=evidence,
        write_uai_file=write_uai_file,
        options=options,
    )
