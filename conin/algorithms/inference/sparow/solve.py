"""Generic solve and solution parsing for the Sparow MAP query models.

The models are built by the ``"sparow"`` representations
(:mod:`conin.representations.sparow`); this module only solves them.
"""

from conin.representations.methods import reject_options_not_taken
import conin.algorithms.inference


def _solve_sparow(repn, q):
    """Solve a MAP query on a ``"sparow"`` representation of an MN, BN or DBN.

    ``q`` is the query of :func:`~conin.algorithms.inference.map_query` (``variables``,
    ``evidence``, ``timing`` and ``options``).  The options are
    ``use_cut_generation`` (omit the c5 constraints and generate them as cuts
    with gurobi) and the options of :func:`solve_pyomo_map_query_model`.

    The solutions are not scored here (``log_potential`` is ``None``): ``map_query``
    scores them.
    """
    options = dict(q.options)
    reject_options_not_taken(
        conin.algorithms.inference.pyomo.solve_pyomo_map_query_model,
        options,
        "map_query(method='integer_program')",
        passed=("timing", "evidence"),
    )
    model = repn.create_model(
        evidence=q.evidence,
        variables=q.variables,
        timing=q.timing,
    )
    return conin.algorithms.inference.pyomo.solve_pyomo_map_query_model(
        model, timing=q.timing, evidence=q.evidence, **options
    )
