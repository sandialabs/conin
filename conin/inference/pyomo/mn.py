"""Pyomo MAP inference for (constrained) Markov networks."""

from conin.constrained_models import ConstrainedDiscreteMarkovNetwork

from .solve import solve_pyomo_map_query_model


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
        Not supported (raises ``RuntimeError``).
    evidence : dict, optional
        ``{node: state}``.
    timing : bool, optional
        Print timing information.
    **options
        ``use_cut_generation`` (omit the c5 constraints and generate them as
        cuts with gurobi) and the options of
        :func:`~conin.inference.pyomo.solve_pyomo_map_query_model`.
    """
    use_cut_generation = options.pop("use_cut_generation", False)

    if variables:  # pragma:nocover
        raise RuntimeError("VariableElimination is not supported for CONIN models")

    cmodel = (
        pgm
        if isinstance(pgm, ConstrainedDiscreteMarkovNetwork)
        else ConstrainedDiscreteMarkovNetwork(pgm)
    )
    model = cmodel.representation("pyomo").create_model(
        evidence=evidence,
        variables=variables,
        omit_c5=use_cut_generation,
        timing=timing,
    )
    return solve_pyomo_map_query_model(
        model, timing=timing, evidence=evidence, **options
    )
