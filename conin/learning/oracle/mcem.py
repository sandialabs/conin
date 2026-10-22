import munch
import math

from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.exceptions import InvalidInputError
from conin.learning.unconstrained.supervised import supervised_learning
from conin.sampling.oracle import rejection_sample_hidden

nan = float("nan")


def mcem(
    # Monte Carlo EM
    #
    # Replace the expectation in the computation of Q(theta | theta_k) with Monte Carlo
    # integration.
    *,
    app,
    observed,
    hidden_states,
    observable_states,
    samples_per_iteration=None,
    max_iterations=None,
    convergence_tolerance=None,
    start_tolerance=None,
    transition_tolerance=None,
    emission_tolerance=None,
):
    """Monte Carlo EM for an HMM with oracle constraints.

    Each iteration samples ``samples_per_iteration`` feasible hidden sequences
    from the current model (rejection sampling, see
    :func:`conin.sampling.oracle.rejection_sample_hidden`), scores each one with
    the current HMM, and re-estimates the HMM from them with
    :func:`~conin.learning.supervised_learning`.  The next iteration samples
    from the re-estimated HMM (``app.with_model(new_hmm)``), with the same
    constraints and data.

    Parameters
    ----------
    app : ConstrainedHiddenMarkovModel or HiddenMarkovModel
        The starting model; a bare HMM is sampled without constraints.  It is
        not changed.
    observed : list
        Observed labels.
    hidden_states, observable_states : list
        Passed to ``supervised_learning``.
    samples_per_iteration, max_iterations, convergence_tolerance : optional
        Defaults 10, 100 and 1e-4.
    start_tolerance, transition_tolerance, emission_tolerance : float, optional
        Passed to ``supervised_learning`` (default 1e-4).

    Returns
    -------
    float
        The average ``log P(observed, hidden)`` of the last iteration's
        samples, under the HMM they were sampled from.
    """
    if len(observed) == 0:
        raise InvalidInputError("No observed states")
    if hidden_states is None or len(hidden_states) == 0:
        raise InvalidInputError("No hidden states specified")
    if observable_states is None or len(observable_states) == 0:
        raise InvalidInputError("No observable states specified")
    if samples_per_iteration is None:
        samples_per_iteration = 10
    if max_iterations is None:
        max_iterations = 100
    if convergence_tolerance is None:
        convergence_tolerance = 1e-4
    if start_tolerance is None:
        start_tolerance = 1e-4
    if transition_tolerance is None:
        transition_tolerance = 1e-4
    if emission_tolerance is None:
        emission_tolerance = 1e-4

    # The re-estimated model is kept locally (a ConstrainedHiddenMarkovModel is
    # immutable) and is not written back to ``app``.
    cmodel = ConstrainedHiddenMarkovModel.ensure(app)

    prev_log_prob = nan
    iteration = 0
    while iteration < max_iterations:
        # Expectation step
        # Sample feasible hidden states
        simulations = []
        log_prob = 0.0
        for _ in range(samples_per_iteration):
            feasible_hidden = rejection_sample_hidden(cmodel, len(observed))
            log_prob = log_prob + cmodel.model.log_probability(
                observed, feasible_hidden
            )
            simulations.append(munch.Munch(observed=observed, hidden=feasible_hidden))

        log_prob /= samples_per_iteration

        # Maximization step
        hmm = supervised_learning(
            simulations=simulations,
            hidden_states=hidden_states,
            observable_states=observable_states,
            start_tolerance=start_tolerance,
            transition_tolerance=transition_tolerance,
            emission_tolerance=emission_tolerance,
        )
        cmodel = cmodel.with_model(hmm)

        if iteration >= max_iterations:
            break
        if not math.isnan(prev_log_prob):
            assert (
                log_prob < prev_log_prob + convergence_tolerance
            ), f"Expecting decreasing log-probabilities: curr={log_prob} prev={prev_log_prob}"
            if abs(log_prob - prev_log_prob) < convergence_tolerance:
                break

        iteration += 1

    return log_prob
