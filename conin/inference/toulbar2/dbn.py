"""toulbar2 MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
)


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
        Passed to the constraint functions as ``data.variables``.
    evidence : dict, optional
        ``{(node, t): state}``.
    write_uai_file : str, optional
        Write the UAI file that toulbar2 reads to this path.
    **options
        ``timing``, ``verbose`` and the options of
        :func:`~conin.inference.toulbar2.solve_toulbar2_map_query_model`.
    """
    if not pytoulbar2_available:
        return _not_available()

    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    model = _create_model(
        cmodel.representation("toulbar2", start=start, stop=stop),
        evidence=evidence,
        variables=variables,
        timing=options.get("timing", False),
        write_uai_file=write_uai_file,
        options=options,
    )
    return solve_toulbar2_map_query_model(model, **options)
