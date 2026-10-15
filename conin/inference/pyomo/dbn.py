"""Pyomo MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork

from .bn import _validate_bn_query
from .solve import solve_pyomo_map_query_model


def inference_pyomo_map_query_DDBN(
    *,
    pgm,
    start=0,
    stop=1,
    variables=None,
    evidence=None,
    **options,
):
    """MAP query on a DBN unrolled over ``start..stop`` with the factor-repn IP.

    Parameters
    ----------
    pgm : DynamicDiscreteBayesianNetwork or ConstrainedDynamicDiscreteBayesianNetwork
    start, stop : int, optional
        The time steps to unroll (structure params of the ``"pyomo"`` representation).
    variables : list, optional
        Only ``None`` or all unrolled nodes are supported.
    evidence : dict, optional
        ``{(node, t): state}``.
    **options
        As for :func:`~conin.inference.pyomo.inference_pyomo_map_query_BN`.
    """
    use_cut_generation = options.pop("use_cut_generation", False)

    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    repn = cmodel.representation("pyomo", start=start, stop=stop)
    _validate_bn_query(options, variables, repn.nodes, evidence=evidence)
    model = repn.create_model(
        evidence=evidence, variables=variables, omit_c5=use_cut_generation
    )
    return solve_pyomo_map_query_model(model, evidence=evidence, **options)
