"""toulbar2 MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork

from .solve import (
    _create_model,
    _not_available,
    pytoulbar2_available,
    solve_toulbar2_map_query_model,
)


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
    variables, evidence, timing, write_uai_file, **options
        As for :func:`~conin.inference.toulbar2.inference_toulbar2_map_query_MN`.
    """
    if not pytoulbar2_available:
        return _not_available()

    cmodel = ConstrainedDiscreteBayesianNetwork.ensure(pgm)
    model = _create_model(
        cmodel.representation("toulbar2"),
        evidence=evidence,
        variables=variables,
        timing=timing,
        write_uai_file=write_uai_file,
        options=options,
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)
