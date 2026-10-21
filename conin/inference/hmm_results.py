"""Shape the results of HMM MAP queries.

The HMM queries return the hidden states as a list, or as a dict keyed by time
step when the observations were given as a dict.
"""


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
