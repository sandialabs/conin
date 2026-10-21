"""HMM evidence (observations) checks and conversions shared by the HMM MAP queries."""

from conin.exceptions import InvalidInputError


def require_hmm_evidence(evidence, what="HMM MAP inference"):
    """Raise ``InvalidInputError`` unless ``evidence`` is a list or a dict.

    The observations of an HMM query are a list, or a dict keyed by time step.
    """
    if type(evidence) is list or type(evidence) is dict:
        return
    raise InvalidInputError(
        f"{what} needs the observations as evidence: a list, or a dict keyed "
        f"by time step (got {type(evidence).__name__})"
    )


def dense_observations(evidence):
    """The observations as a list (a dict is read at the times ``0..len - 1``)."""
    if type(evidence) is dict:
        return [evidence[i] for i in range(len(evidence))]
    return evidence


def key_solutions_by_time(results):
    """Make the ``states``/``hidden`` of every solution a dict keyed by time step."""
    for soln in results.solutions:
        soln.states = {i: v for i, v in enumerate(soln.states)}
        soln.hidden = soln.states
    return results


def hmm_dbn_evidence(evidence, start, stop):
    """HMM observations -> evidence ``{("E", t): obs}`` of the unrolled DBN."""
    if type(evidence) is list:
        return {("E", i): evidence[i] for i in range(start, stop + 1)}
    return {("E", k): v for k, v in evidence.items()}


def hmm_states_from_dbn(results, evidence, start, stop):
    """Set ``solution.states`` to the hidden states (a list, or a dict for dict evidence)."""
    if type(evidence) is list:
        results.solution.states = [
            results.solution.states["H", i] for i in range(start, stop + 1)
        ]
    else:
        results.solution.states = {
            i: results.solution.states["H", i] for i in range(start, stop + 1)
        }
    return results
