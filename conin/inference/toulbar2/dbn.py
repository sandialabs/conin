"""toulbar2 MAP inference for (constrained) dynamic Bayesian networks."""

from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
from conin.inference.query_checks import map_query_args

from .solve import _solve_toulbar2


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
    timing = options.pop("timing", False)
    cmodel = ConstrainedDynamicDiscreteBayesianNetwork.ensure(pgm)
    return _solve_toulbar2(
        cmodel.representation("toulbar2", start=start, stop=stop),
        map_query_args(
            variables=variables,
            evidence=evidence,
            timing=timing,
            options=dict(options, write_uai_file=write_uai_file),
        ),
    )
