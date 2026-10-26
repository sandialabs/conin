"""Pyomo MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from conin.inference.query_checks import map_query_args

from .solve import _solve_pyomo


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
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    return _solve_pyomo(
        cmodel.representation("pyomo", start=start, stop=stop),
        map_query_args(variables=variables, evidence=evidence, options=options),
    )
