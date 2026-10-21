"""toulbar2 MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.inference._query_checks import validate_bn_query

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
)


def _map_query_toulbar2_bn(
    repn, *, variables, evidence, timing, write_uai_file, options
):
    """Solve a MAP query on the ``"toulbar2"`` representation of a Bayesian network."""
    if not pytoulbar2_available:
        return _not_available()
    variables = validate_bn_query(options, variables, repn.nodes, evidence=evidence)
    model = _create_model(
        repn,
        evidence=evidence,
        variables=variables,
        timing=timing,
        write_uai_file=write_uai_file,
        options=options,
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)


def inference_toulbar2_map_query_BN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    write_uai_file=None,
    **options,
):
    """MAP query on a Bayesian network with toulbar2.

    The CFN is read from a BAYES UAI file (toulbar2 is assumed to handle
    BAYES and MARKOV files appropriately).

    Parameters
    ----------
    pgm : DiscreteBayesianNetwork or ConstrainedDiscreteBayesianNetwork
    variables : list, optional
        Only ``None`` or all nodes are supported (``RuntimeError``, as the
        integer-program query).
    evidence, timing, write_uai_file
        As for :func:`~conin.inference.toulbar2.inference_toulbar2_map_query_MN`.
    **options
        ``prune_network`` and ``create_MN`` (both unsupported: ``True`` raises
        ``RuntimeError``), ``verbose`` and the options of
        :func:`~conin.inference.toulbar2.solve_toulbar2_map_query_model`.
    """
    if not pytoulbar2_available:
        return _not_available()

    cmodel = ConstrainedDiscreteBayesianNetwork.ensure(pgm)
    return _map_query_toulbar2_bn(
        cmodel.representation("toulbar2"),
        variables=variables,
        evidence=evidence,
        timing=timing,
        write_uai_file=write_uai_file,
        options=options,
    )
