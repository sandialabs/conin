"""Pyomo MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.inference.query_checks import map_query_args

from .solve import _solve_pyomo


def inference_pyomo_map_query_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    """MAP query on a Bayesian network with the factor-repn integer program.

    The IP is built on the BN's Markov-network skeleton (one factor per CPD).

    Parameters
    ----------
    pgm : DiscreteBayesianNetwork or ConstrainedDiscreteBayesianNetwork
    variables : list, optional
        Only ``None`` or all nodes are supported.
    evidence : dict, optional
        ``{node: state}``.
    timing : bool, optional
        Print timing information.
    **options
        ``use_cut_generation``, ``prune_network`` and ``create_MN`` (both
        unsupported: ``True`` raises ``RuntimeError``), and the options of
        :func:`~conin.inference.pyomo.solve_pyomo_map_query_model`.
    """
    cmodel = ConstrainedDiscreteBayesianNetwork.ensure(pgm)
    return _solve_pyomo(
        cmodel.representation("pyomo"),
        map_query_args(
            variables=variables, evidence=evidence, timing=timing, options=options
        ),
    )
