"""Rejection sampling of feasible hidden-state sequences."""

from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel
from conin.algorithms.sampling.unconstrained import generate_hidden
from conin.algorithms.sampling.unconstrained.hmm import _hidden_indices


def rejection_sample_hidden(hmm, time_steps, *, max_failures=1000):
    """Sample a hidden-state sequence that satisfies the oracle constraints.

    Sequences are drawn from the unconstrained HMM (with NumPy's global random
    state, see ``HiddenMarkovModel.set_seed``) until one is feasible.

    Parameters
    ----------
    hmm : ConstrainedHiddenMarkovModel or HiddenMarkovModel
        The model; a bare HMM is sampled without constraints.  Its constraints
        must have an ``"oracle"`` representation (no constraints or
        ``"oracle"`` constraints).
    time_steps : int
        Length of the sequence.
    max_failures : int, optional
        Number of rejected samples allowed: the search gives up when one more
        sample is rejected (at most ``max_failures + 1`` samples are drawn).

    Returns
    -------
    list
        Feasible sequence of external hidden-state labels.

    Raises
    ------
    RuntimeError
        If more than ``max_failures`` samples are rejected.
    InvalidInputError
        If ``time_steps`` is negative, ``hmm`` is not an HMM, or the
        constraint family has no ``"oracle"`` representation.
    """
    if isinstance(hmm, HiddenMarkovModel):
        # No constraints: every sample is feasible (same random draws).
        return generate_hidden(hmm, time_steps)
    if not hasattr(hmm, "representation") or not isinstance(
        getattr(hmm, "model", None), HiddenMarkovModel
    ):
        raise InvalidInputError(
            "rejection_sample_hidden expects a HiddenMarkovModel or a "
            f"ConstrainedHiddenMarkovModel, got {type(hmm).__name__}"
        )
    repn = hmm.representation("oracle")
    start = repn.hmm.start_vec.tolist()
    transition = repn.hmm.transition_mat.tolist()
    h2e = repn.model.hidden_to_external
    for _ in range(max_failures + 1):
        hidden = _hidden_indices(start, transition, time_steps)
        if repn.is_feasible(hidden):
            return [h2e[h] for h in hidden]
    raise RuntimeError(
        "Failed to generate a feasible hidden state: "
        f"{max_failures + 1} samples were rejected"
    )
