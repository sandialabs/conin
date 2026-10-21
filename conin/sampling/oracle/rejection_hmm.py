"""Rejection sampling of feasible hidden-state sequences (from ``Oracle_CHMM.generate_hidden``)."""

from conin.exceptions import InvalidInputError
from conin.graphical_models import HiddenMarkovModel


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
        Number of rejected samples after which the search gives up.

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
        return hmm.generate_hidden(time_steps)
    if not hasattr(hmm, "representation") or not isinstance(
        getattr(hmm, "model", None), HiddenMarkovModel
    ):
        raise InvalidInputError(
            "rejection_sample_hidden expects a HiddenMarkovModel or a "
            f"ConstrainedHiddenMarkovModel, got {type(hmm).__name__}"
        )
    repn = hmm.representation("oracle")
    hidden = repn.hmm.generate_hidden(time_steps)
    ctr = 0
    while not repn.is_feasible(hidden):
        hidden = repn.hmm.generate_hidden(time_steps)
        ctr += 1
        if ctr > max_failures:
            raise RuntimeError(
                f"Failed to generate a feasible hidden state after {max_failures} trials"
            )
    h2e = repn.model.hidden_to_external
    return [h2e[h] for h in hidden]
