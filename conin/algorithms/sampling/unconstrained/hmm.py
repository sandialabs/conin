"""Forward sampling of hidden Markov models without constraints.

Every sampler takes ``rng``: ``None`` (a new generator seeded from the
operating system), a seed, or a :class:`numpy.random.Generator` (see
:func:`conin.util.as_rng`).  Pass the same generator to successive calls to
continue its stream.
"""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.exceptions import InvalidInputError
from conin.util import Util, as_rng


def _arrays(hmm):
    """Return ``(model, start, transition, emission)`` as lists on internal indices."""
    cmodel = ConstrainedHiddenMarkovModel.ensure(hmm)
    # Raises InvalidInputError if the model has constraints (no "matvec"
    # representation), so constraints are never silently ignored.
    repn = cmodel.representation("matvec")
    return (
        cmodel.model,
        repn.start_vec.tolist(),
        repn.transition_mat.tolist(),
        repn.emission_mat.tolist(),
    )


def _hidden_indices(start, transition, time_steps, rng):
    """Sample ``time_steps`` hidden-state indices (none for ``time_steps == 0``)."""
    if time_steps < 0:
        raise InvalidInputError(f"time_steps must be non-negative, got {time_steps}")
    if time_steps == 0:
        return []
    hidden = [Util.sample_from_vec(start, rng)]
    for t in range(time_steps - 1):
        hidden.append(Util.sample_from_vec(transition[hidden[t]], rng))
    return hidden


def _observed_indices(emission, hidden, rng):
    """Sample one observed-state index per hidden-state index."""
    return [Util.sample_from_vec(emission[h], rng) for h in hidden]


def generate_hidden(hmm, time_steps, *, rng=None):
    """Sample a sequence of hidden-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model; a constrained model must have no constraints (use
        :func:`conin.algorithms.sampling.oracle.rejection_sample_hidden` for oracle
        constraints).
    time_steps : int
        Number of hidden states to generate.
    rng : None, int or numpy.random.Generator, optional
        The random generator, or a seed for a new one (see
        :func:`conin.util.as_rng`).

    Returns
    -------
    list
        Generated hidden-state labels.

    Raises
    ------
    InvalidInputError
        If ``time_steps`` is negative or the model has constraints.
    """
    model, start, transition, _ = _arrays(hmm)
    h2e = model.hidden_to_external
    hidden = _hidden_indices(start, transition, time_steps, as_rng(rng))
    return [h2e[h] for h in hidden]


def generate_hidden_until_state(hmm, h, *, max_steps=10_000, rng=None):
    """Sample hidden-state labels until a target state is reached.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    h : Any
        Target hidden-state label at which sampling stops.
    max_steps : int, optional
        The longest sequence to sample (default 10000).  The target may be
        unreachable (e.g. after an absorbing state), so sampling gives up
        after ``max_steps`` states.
    rng : None, int or numpy.random.Generator, optional
        The random generator, or a seed for a new one (see
        :func:`conin.util.as_rng`).

    Returns
    -------
    list
        Generated hidden-state labels ending in ``h`` (at most
        ``max_steps``).

    Raises
    ------
    InvalidInputError
        If ``max_steps`` is less than 1 or the model has constraints.
    RuntimeError
        If ``h`` is not reached within ``max_steps`` states.
    """
    if max_steps < 1:
        raise InvalidInputError(f"max_steps must be at least 1, got {max_steps}")
    model, start, transition, _ = _arrays(hmm)
    target = model.hidden_to_internal[h]
    rng = as_rng(rng)
    hidden = [Util.sample_from_vec(start, rng)]
    while hidden[-1] != target:
        if len(hidden) == max_steps:
            raise RuntimeError(
                f"Failed to reach the hidden state {h!r} within "
                f"max_steps={max_steps} states"
            )
        hidden.append(Util.sample_from_vec(transition[hidden[-1]], rng))
    h2e = model.hidden_to_external
    return [h2e[i] for i in hidden]


def generate_observed_from_hidden(hmm, hidden, *, rng=None):
    """Sample observed-state labels given hidden-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    hidden : list
        Hidden-state labels from which to generate observations.
    rng : None, int or numpy.random.Generator, optional
        The random generator, or a seed for a new one (see
        :func:`conin.util.as_rng`).

    Returns
    -------
    list
        Generated observed-state labels, one per hidden state.
    """
    model, _, _, emission = _arrays(hmm)
    h2i = model.hidden_to_internal
    o2e = model.observed_to_external
    observed = _observed_indices(emission, [h2i[h] for h in hidden], as_rng(rng))
    return [o2e[o] for o in observed]


def generate_observed(hmm, time_steps, *, rng=None):
    """Sample a hidden-state sequence and return its observed-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    time_steps : int
        Number of time steps for which to generate observations.
    rng : None, int or numpy.random.Generator, optional
        The random generator, or a seed for a new one (see
        :func:`conin.util.as_rng`).

    Returns
    -------
    list
        Generated observed-state labels.

    Raises
    ------
    InvalidInputError
        If ``time_steps`` is negative or the model has constraints.
    """
    model, start, transition, emission = _arrays(hmm)
    rng = as_rng(rng)
    hidden = _hidden_indices(start, transition, time_steps, rng)
    o2e = model.observed_to_external
    return [o2e[o] for o in _observed_indices(emission, hidden, rng)]
