"""Solve a toulbar2 MAP query model, and MAP inference for (constrained) Markov networks.

The models are built by the ``"toulbar2"`` representations
(:mod:`conin.representations.toulbar2`).
"""

import munch
from pyomo.common.timing import TicTocTimer

from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2

_NOT_AVAILABLE = "pytoulbar2 not available"


def _not_available():
    return munch.Munch(
        solution=None,
        solutions=[],
        termination_condition=_NOT_AVAILABLE,
        solvetime=0.0,
    )


def _create_model(repn, *, evidence, variables, timing, write_uai_file, options):
    """Pop the per-query ``verbose`` option and build a fresh CFN from ``repn``."""
    verbose = options.pop("verbose", -1)
    return repn.create_model(
        evidence=evidence,
        variables=variables,
        verbose=verbose,
        write_uai_file=write_uai_file,
        timing=timing,
    )


def solve_toulbar2_map_query_model(
    model,
    *,
    solution_with_fixed=False,
    solution_with_evidence=False,
    timing=False,
):
    solution_with_evidence = solution_with_fixed or solution_with_evidence
    if timing:  # pragma:nocover
        timer = TicTocTimer()
        timer.tic("CFN_map_query - START")

    solver_timer = TicTocTimer()
    solver_timer.tic(None)
    res = model.Solve()
    solvetime = solver_timer.toc(None)
    solution, primal_bound, num_solutions = res
    var = {
        name: model.states[i][solution[i]]
        for name, i in model.V.items()
        if solution_with_evidence or name not in model.V_evidence
    }
    soln = munch.Munch(states=var, log_factor_sum=None, primal_bound=primal_bound)

    if timing:  # pragma:nocover
        timer.toc("Completed optimization")

    return munch.Munch(
        solution=soln,
        solutions=[soln],
        termination_condition="ok",
        solvetime=solvetime,
    )


def inference_toulbar2_map_query_MN(
    *,
    pgm,
    variables=None,
    evidence=None,
    timing=False,
    write_uai_file=None,
    **options,
):
    """MAP query on a Markov network with toulbar2.

    Parameters
    ----------
    pgm : DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork
    variables : list, optional
        Passed to the constraint functions as ``data.variables``.
    evidence : dict, optional
        ``{node: state}``.
    timing : bool, optional
        Print timing information.
    write_uai_file : str, optional
        Write the UAI file that toulbar2 reads to this path.
    **options
        ``verbose`` (the toulbar2 verbosity, default ``-1``) and the options of
        :func:`solve_toulbar2_map_query_model`.
    """
    if not pytoulbar2_available:  # pragma:nocover
        return _not_available()

    cmodel = ConstrainedDiscreteMarkovNetwork.ensure(pgm)
    model = _create_model(
        cmodel.representation("toulbar2"),
        evidence=evidence,
        variables=variables,
        timing=timing,
        write_uai_file=write_uai_file,
        options=options,
    )
    return solve_toulbar2_map_query_model(model, timing=timing, **options)
