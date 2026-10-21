"""Pyomo MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.inference.query_checks import full_query_variables, reject_pruning_options

from .solve import solve_pyomo_map_query_model


def _map_query_pyomo_bn(repn, *, variables, evidence, timing, options):
    """Solve a MAP query on the ``"pyomo"`` representation of a Bayesian network."""
    use_cut_generation = options.pop("use_cut_generation", False)
    reject_pruning_options(options)
    variables = full_query_variables(variables, repn.nodes)
    model = repn.create_model(
        evidence=evidence,
        variables=variables,
        omit_c5=use_cut_generation,
        timing=timing,
    )
    return solve_pyomo_map_query_model(
        model, timing=timing, evidence=evidence, **options
    )


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
    return _map_query_pyomo_bn(
        cmodel.representation("pyomo"),
        variables=variables,
        evidence=evidence,
        timing=timing,
        options=options,
    )
