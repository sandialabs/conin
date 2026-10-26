"""The query methods and the representation kind each one runs on.

One table for :func:`~conin.inference.map_query` and
:func:`~conin.scoring.log_probability_query`: a method that both queries
support (``"variable_elimination"``) runs on the same representation for
both.  Which (model type, constraint family) pairs a method supports is
decided by the representation registry, not here.
"""


def _integer_program_kind(cmodel, options):
    """``ip_formulation`` (popped from ``options``) selects the formulation."""
    ip_formulation = options.pop("ip_formulation", None)
    if ip_formulation is None or ip_formulation == "markov_network":
        return "pyomo"
    if ip_formulation == "network_flow":
        return "pyomo_network_flow"
    raise ValueError(f"Unexpected ip_formulation value: '{ip_formulation}'")


def _a_star_kind(cmodel, options):
    """Recursive constraints use recursive A*, the others A* with oracles."""
    return "recursive" if cmodel.constraint_family == "recursive" else "oracle"


REPRESENTATION_KINDS = {
    "integer_program": _integer_program_kind,
    "toulbar2": "toulbar2",
    "variable_elimination": "pgmpy",
    "a_star": _a_star_kind,
    "viterbi": "matvec",
    "mvr_viterbi": "mvr",
}
"""Method -> representation kind, or a function ``(cmodel, options) -> kind``."""

MAP_METHODS = (
    "integer_program",
    "toulbar2",
    "a_star",
    "viterbi",
    "variable_elimination",
    "mvr_viterbi",
)
"""The ``method`` values of :func:`~conin.inference.map_query`."""

LOG_PROBABILITY_METHODS = ("variable_elimination",)
"""The ``method`` values of :func:`~conin.scoring.log_probability_query`."""


def check_method(method, methods, what):
    """Raise ``ValueError("Unsupported <what>: ...")`` unless ``method`` is in ``methods``."""
    if method not in methods:
        raise ValueError(
            f"Unsupported {what}: {method}. Expected one of: {', '.join(methods)}"
        )


def representation_kind(method, cmodel, options):
    """Return the representation kind that ``method`` runs on for ``cmodel``.

    ``options`` are the query's method options; a method may pop the ones
    that select the kind (``ip_formulation`` for ``"integer_program"``).

    Raises
    ------
    ValueError
        For an unknown ``ip_formulation``.
    """
    kind = REPRESENTATION_KINDS[method]
    return kind(cmodel, options) if callable(kind) else kind
