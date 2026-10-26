"""Shape the results of HMM MAP queries.

The HMM queries return the hidden states as a list, or as a dict keyed by time
step when the observations were given as a dict.
"""

import munch

from conin.graphical_models.hmm_observations import unrolled_evidence


def key_solutions_by_time(results):
    """Make the ``states``/``hidden`` of every solution a dict keyed by time step."""
    for soln in results.solutions:
        soln.states = {i: v for i, v in enumerate(soln.states)}
        soln.hidden = soln.states
    return results


def hidden_states_from_unrolled(results, *, start, stop, keyed_by_time):
    """Set ``solution.states`` to the hidden states ``("H", t)`` of the unrolled DBN.

    The states at the times ``start..stop`` become a list, or a dict keyed by
    time step if ``keyed_by_time``.
    """
    states = results.solution.states
    times = range(start, stop + 1)
    if keyed_by_time:
        results.solution.states = {t: states["H", t] for t in times}
    else:
        results.solution.states = [states["H", t] for t in times]
    return results


def solve_unrolled_hmm(solve, repn, q):
    """Solve an HMM query as a query on its unrolled DBN.

    ``repn`` is the representation of an HMM unrolled over
    ``repn.start..repn.stop`` and ``solve(repn, q)`` the solver of its
    backend.  The observations ``q.observed`` become the evidence
    ``{("E", t): o}`` of the unrolled DBN, and the hidden states ``("H", t)``
    of the solution become ``solution.states`` (a dict keyed by time step if
    the evidence ``q.evidence`` was a dict).  A result without a solution is
    returned as it is.
    """
    start, stop = repn.start, repn.stop
    dbn_query = munch.Munch(q, evidence=unrolled_evidence(q.observed, start, stop))
    results = solve(repn, dbn_query)
    if results.solution is None:
        return results
    return hidden_states_from_unrolled(
        results, start=start, stop=stop, keyed_by_time=isinstance(q.evidence, dict)
    )
