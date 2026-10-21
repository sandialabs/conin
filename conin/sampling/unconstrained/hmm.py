"""Forward sampling of hidden Markov models without constraints.

The samplers use NumPy's global random state (see
``HiddenMarkovModel.set_seed``).
"""

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.exceptions import InvalidInputError
from conin.util import Util


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


def _hidden_indices(start, transition, time_steps):
    """Sample ``time_steps`` hidden-state indices."""
    if time_steps < 0:
        raise InvalidInputError("In generate_hidden time_steps > 0.")
    hidden = [Util.sample_from_vec(start)]
    for t in range(time_steps - 1):
        hidden.append(Util.sample_from_vec(transition[hidden[t]]))
    return hidden


def _observed_indices(emission, hidden):
    """Sample one observed-state index per hidden-state index."""
    return [Util.sample_from_vec(emission[h]) for h in hidden]


def generate_hidden(hmm, time_steps):
    """Sample a sequence of hidden-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model; a constrained model must have no constraints (use
        :func:`conin.sampling.oracle.rejection_sample_hidden` for oracle
        constraints).
    time_steps : int
        Number of hidden states to generate.

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
    return [h2e[h] for h in _hidden_indices(start, transition, time_steps)]


def generate_hidden_until_state(hmm, h):
    """Sample hidden-state labels until a target state is reached.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    h : Any
        Target hidden-state label at which sampling stops.

    Returns
    -------
    list
        Generated hidden-state labels ending in ``h``.
    """
    model, start, transition, _ = _arrays(hmm)
    target = model.hidden_to_internal[h]
    hidden = [Util.sample_from_vec(start)]
    while hidden[-1] != target:
        hidden.append(Util.sample_from_vec(transition[hidden[-1]]))
    h2e = model.hidden_to_external
    return [h2e[i] for i in hidden]


def generate_observed_from_hidden(hmm, hidden):
    """Sample observed-state labels given hidden-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    hidden : list
        Hidden-state labels from which to generate observations.

    Returns
    -------
    list
        Generated observed-state labels, one per hidden state.
    """
    model, _, _, emission = _arrays(hmm)
    h2i = model.hidden_to_internal
    o2e = model.observed_to_external
    return [o2e[o] for o in _observed_indices(emission, [h2i[h] for h in hidden])]


def generate_observed(hmm, time_steps):
    """Sample a hidden-state sequence and return its observed-state labels.

    Parameters
    ----------
    hmm : HiddenMarkovModel or ConstrainedHiddenMarkovModel
        The model (without constraints).
    time_steps : int
        Number of time steps for which to generate observations.

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
    if time_steps < 0:
        raise InvalidInputError("In generate_observed time_steps > 0.")
    hidden = _hidden_indices(start, transition, time_steps)
    o2e = model.observed_to_external
    return [o2e[o] for o in _observed_indices(emission, hidden)]
