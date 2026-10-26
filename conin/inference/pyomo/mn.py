"""Pyomo MAP inference for (constrained) Markov networks."""

from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.inference.query_checks import map_query_args

from .solve import _solve_pyomo


def inference_pyomo_map_query_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    **options,
):
    """MAP query on a Markov network with the factor-repn integer program.

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork
    variables : list, optional
        Only ``None`` or all nodes are supported (``RuntimeError``).
    evidence : dict, optional
        ``{node: state}``.
    timing : bool, optional
        Print timing information.
    **options
        ``use_cut_generation`` (omit the c5 constraints and generate them as
        cuts with gurobi) and the options of
        :func:`~conin.inference.pyomo.solve_pyomo_map_query_model`.
    """
    cmodel = ConstrainedDiscreteMarkovNetwork.ensure(pgm)
    return _solve_pyomo(
        cmodel.representation("pyomo"),
        map_query_args(
            variables=variables, evidence=evidence, timing=timing, options=options
        ),
    )
