"""Solve a toulbar2 MAP query model, and MAP inference for (constrained) Markov networks.

The models are built by the ``"toulbar2"`` representations
(:mod:`conin.representations.toulbar2`).
"""

import munch
from pyomo.common.timing import TicTocTimer

from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.inference.query_checks import (
    full_query_variables,
    map_query_args,
    reject_pruning_options,
)
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


def _solve_toulbar2(repn, q):
    """Solve a MAP query on a ``"toulbar2"`` representation of an MN, BN or DBN.

    ``q`` is the query of :func:`~conin.inference.map_query` (``variables``,
    ``evidence``, ``timing`` and ``options``).  The options are ``verbose``
    (the toulbar2 verbosity, default ``-1``), ``write_uai_file`` (write the
    UAI file that toulbar2 reads to this path), ``prune_network`` and
    ``create_MN`` (``True`` raises ``RuntimeError``), and the options of
    :func:`solve_toulbar2_map_query_model`.  An HMM is solved as its unrolled
    DBN (:func:`~conin.inference.hmm_results.solve_unrolled_hmm`).
    """
    if not pytoulbar2_available:
        return _not_available()
    options = dict(q.options)
    verbose = options.pop("verbose", -1)
    write_uai_file = options.pop("write_uai_file", None)
    reject_pruning_options(options)
    model = repn.create_model(
        evidence=q.evidence,
        variables=full_query_variables(q.variables, repn.nodes),
        verbose=verbose,
        write_uai_file=write_uai_file,
        timing=q.timing,
    )
    return solve_toulbar2_map_query_model(model, timing=q.timing, **options)


def solve_toulbar2_map_query_model(
    model,
    *,
    solution_with_fixed=False,
    solution_with_evidence=False,
    timing=False,
):
    """Solve a toulbar2 MAP query model.

    Parameters
    ----------
    model : pytoulbar2.CFN
        A model built by the ``create_model()`` method of a ``"toulbar2"``
        representation (it has the ``V``, ``states`` and ``V_evidence``
        attributes that map the CFN variables back to the model's nodes).
    solution_with_fixed, solution_with_evidence : bool, optional
        Include the evidence variables in the solution (the two names are
        synonyms).
    timing : bool, optional
        Print timing information.

    Returns
    -------
    munch.Munch
        ``solution`` (with ``states``, ``{node: state}``, and the CFN
        ``primal_bound``), ``solutions`` (``[solution]``),
        ``termination_condition`` (``"ok"``) and ``solvetime``.
    """
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
        Not supported (raises ``RuntimeError``, as the integer-program query).
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
    cmodel = ConstrainedDiscreteMarkovNetwork.ensure(pgm)
    return _solve_toulbar2(
        cmodel.representation("toulbar2"),
        map_query_args(
            variables=variables,
            evidence=evidence,
            timing=timing,
            options=dict(options, write_uai_file=write_uai_file),
        ),
    )
