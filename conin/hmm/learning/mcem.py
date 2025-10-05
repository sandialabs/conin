from collections import deque
import munch

# from conin.hmm import HiddenMarkovModel
from .supervised import supervised_learning
import math

nan = float("nan")


def mcem(
    # Monte Carlo EM
    #
    # Replace the expectation in the computation of Q(theta | theta_k) with Monte Carlo
    # integration.
    *,
    app,
    observations,
    hidden_states,
    observable_states,
    samples_per_iteration=None,
    max_iterations=None,
    convergence_tolerance=None,
    start_tolerance=None,
    transition_tolerance=None,
    emission_tolerance=None,
):
    assert len(observations) > 0, "No observations"
    assert (
        hidden_states is not None and len(hidden_states) != 0
    ), "No hidden states specified"
    assert (
        observable_states is not None and len(observable_states) != 0
    ), "No observable states specified"
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

    prev_log_prob = nan
    iteration = 0
    while iteration < max_iterations:
        # Expectation step
        # Sample feasible hidden states
        simulations = []
        log_prob = 0.0
        for _ in range(samples_per_iteration):
            feasible_hidden = app.generate_hidden(observations)
            log_prob = log_prob + app.hmm.log_probability(observations, feasible_hidden)
            simulations.append(
                munch.Munch(
                    observations=observations,
                    hidden=app.generate_hidden(observations),
                )
            )

        log_prob /= samples_per_iteration

        # Maximization step
        app.hmm = supervised_learning(
            simulations=simulations,
            hidden_states=hidden_states,
            observable_states=observable_states,
            start_tolerance=start_tolerance,
            transition_tolerance=transition_tolerance,
            emission_tolerance=emission_tolerance,
        )

        if iteration >= max_iterations:
            break
        if prev_log_prob != nan:
            assert (
                log_prob < prev_log_prob + convergence_tolerance
            ), f"Expecting decreasing log-probabilities: curr={log_prob} prev={prev_log_prob}"
            if math.abs(log_prob - prev_log_prob) < convergence_tolerance:
                break

        iteration += 1

    return log_prob
