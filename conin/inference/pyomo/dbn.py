"""Pyomo MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork

from .bn import check_bn_variables, pop_bn_build_options
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
    create_MN = pop_bn_build_options(options)

    cmodel = (
        pgm
        if isinstance(pgm, ConstrainedDynamicDiscreteBayesianNetwork)
        else ConstrainedDynamicDiscreteBayesianNetwork(pgm)
    )
    repn = cmodel.representation("pyomo", start=start, stop=stop)
    check_bn_variables(variables, repn.nodes, evidence=evidence, create_MN=create_MN)
    model = repn.create_model(
        evidence=evidence, variables=variables, omit_c5=use_cut_generation
    )
    return solve_pyomo_map_query_model(model, evidence=evidence, **options)
