"""toulbar2 MAP inference for (constrained) Bayesian networks."""

from conin.constrained_models import ConstrainedDiscreteBayesianNetwork
from conin.inference.query_checks import map_query_args

from .solve import _solve_toulbar2


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
    cmodel = ConstrainedDiscreteBayesianNetwork.ensure(pgm)
    return _solve_toulbar2(
        cmodel.representation("toulbar2"),
        map_query_args(
            variables=variables,
            evidence=evidence,
            timing=timing,
            options=dict(options, write_uai_file=write_uai_file),
        ),
    )
