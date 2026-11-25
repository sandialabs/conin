"""The query methods and the representation kind each one runs on.

One table for :func:`~conin.algorithms.inference.map_query` and
:func:`~conin.algorithms.scoring.log_probability_query`: a method that both queries
support (``"variable_elimination"``) runs on the same representation for
both.  Which (model type, constraint family) pairs a method supports is
decided by the representation registry, not here.
"""

import inspect

from conin.exceptions import InvalidInputError


def _integer_program_kind(cmodel, inference_type, options):
    """``ip_formulation`` (popped from ``options``) selects the formulation."""
    ip_formulation = options.pop("ip_formulation", None)
    if inference_type == "mpe_inference":
        if ip_formulation is None or ip_formulation == "markov_network":
            return "pyomo"
        if ip_formulation == "network_flow":
            return "pyomo_network_flow"
        raise InvalidInputError(f"Unexpected ip_formulation value: '{ip_formulation}'")
    elif inference_type == "map_inference":
        if ip_formulation is None or ip_formulation == "markov_network":
            return "sparow"
        if ip_formulation == "network_flow":
            return "sparow_network_flow"
        raise InvalidInputError(f"Unexpected ip_formulation value: '{ip_formulation}'")
    raise InvalidInputError(f"Unexpected inference_type value: '{inference_type}'")


def _a_star_kind(cmodel, inference_type, options):
    """Recursive constraints use recursive A*, the others A* with oracles."""
    return "recursive" if "recursive" in cmodel.constraint_families else "oracle"


REPRESENTATION_KINDS = {
    "integer_program": _integer_program_kind,
    "toulbar2": "toulbar2",
    "variable_elimination": "variable_elimination",
    "a_star": _a_star_kind,
    "viterbi": "matvec",
    "mvr_viterbi": "mvr",
}
"""Method -> representation kind, or a function ``(cmodel, inference_type, options) -> kind``."""

MAP_METHODS = (
    "integer_program",
    "toulbar2",
    "a_star",
    "viterbi",
    "variable_elimination",
    "mvr_viterbi",
)
"""The ``method`` values of :func:`~conin.algorithms.inference.map_query`."""

LOG_PROBABILITY_METHODS = ("variable_elimination",)
"""The ``method`` values of :func:`~conin.algorithms.scoring.log_probability_query`."""


def check_method(method, methods, what):
    """Raise ``InvalidInputError("Unsupported <what>: ...")`` unless ``method`` is in ``methods``."""
    if method not in methods:
        raise InvalidInputError(
            f"Unsupported {what}: {method}. Expected one of: {', '.join(methods)}"
        )


def reject_unexpected_options(options, what):
    """Raise ``TypeError`` if ``options`` (the options a method did not use) is not empty.

    Every method rejects the options it does not know, so that a misspelt
    option is not silently ignored.  ``what`` names the call in the message,
    e.g. ``"map_query(method='viterbi')"``.
    """
    if options:
        raise TypeError(
            f"Unexpected options for {what}: {', '.join(map(str, options))}"
        )


def reject_options_not_taken(func, options, what, *, passed=()):
    """Raise ``TypeError`` for the ``options`` that ``func`` does not take.

    For a method that hands its options to ``func`` as keyword arguments:
    the options are the keyword-only parameters of ``func`` (its positional
    parameters are the arguments that the method itself passes), less
    ``passed``, the keywords that the method sets itself.  The other options
    are rejected with the message of :func:`reject_unexpected_options` rather
    than with Python's message, which names ``func``.
    """
    parameters = inspect.signature(func).parameters
    keywords = {name for name, p in parameters.items() if p.kind is p.KEYWORD_ONLY}
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return
    reject_unexpected_options(
        [name for name in options if name not in keywords or name in passed], what
    )


def representation_kind(method, cmodel, inference_type, options):
    """Return the representation kind that ``method`` runs on for ``cmodel``.

    ``options`` are the query's method options; a method may pop the ones
    that select the kind (``ip_formulation`` for ``"integer_program"``).

    Raises
    ------
    InvalidInputError
        For an unknown ``ip_formulation``.
    """
    kind = REPRESENTATION_KINDS[method]
    return kind(cmodel, inference_type, options) if callable(kind) else kind
