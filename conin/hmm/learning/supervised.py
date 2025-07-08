from conin.hmm import HMM

# TODO also allows for hidden and observed instead of simulations


def supervised_learning(
    # The training data may not exhibit all feasible starting values, so we allow
    # for a default non-zero start_tolerance, transition_tolerance and emission_tolerance.
    *,
    simulations,
    hidden_states,
    observable_states,
    start_tolerance=None,
    transition_tolerance=None,
    emission_tolerance=None,
    transition_prior=None,  # Nonzero values
    emission_prior=None  # Nonzero values
):
    """
    Inputs:
        - simulations: Generated using helper.convert_to_simulations
        - hidden_states: Hidden states that you could possibly have (you don't need to see all of them in the simulations)
        - observable_states: Same as hidden states but for observations
        - X_tolerance: If in a simulation we have a zero in our calculation we replace it with this tolerance
        - X_priors: These are only used for transitions or emissions which are not observed and for which we want some default values
    """
    assert (
        hidden_states is not None and len(hidden_states) != 0
    ), "No hidden states specified"
    assert (
        observable_states is not None and len(observable_states) != 0
    ), "No observable states specified"
    if start_tolerance is None:
        start_tolerance = 1e-4
    if transition_tolerance is None:
        transition_tolerance = 1e-4
    if emission_tolerance is None:
        emission_tolerance = 1e-4
    #
    # Estimate starting probabilities
    #
    start_probs = {i: start_tolerance for i in hidden_states}
    for sim in simulations:
        start_probs[sim.hidden[0]] += 1

    total = sum(start_probs.values())
    if total == 0.0:  # pragma: no cover
        raise ValueError("YOU SHOULD NOT SEE THIS.")
    else:
        for i in start_probs:
            start_probs[i] /= total

    #
    # Estimate transition probabilities
    #
    transition_probs = {
        (i, j): transition_tolerance
        for i in hidden_states
        for j in hidden_states
    }
    rowsum = {
        i: transition_tolerance * len(hidden_states) for i in hidden_states
    }
    for sim in simulations:
        for i, curr in enumerate(sim.hidden):
            if i == 0:
                continue
            prev = sim.hidden[i - 1]
            transition_probs[prev, curr] += 1
            rowsum[prev] += 1

    for i in hidden_states:
        if rowsum[i] == 0.0:
            if transition_prior:
                for j in hidden_states:
                    transition_probs[i, j] = transition_prior.get((i, j), 0.0)
            else:
                # Uniform transition probabilities if we saw no transitions
                # from state i
                for j in hidden_states:
                    transition_probs[i, j] = 1.0 / len(hidden_states)
        else:
            # Normalize transition probabilities from state i
            for j in hidden_states:
                transition_probs[i, j] /= rowsum[i]

    #
    # Estimate emission probabilities
    #
    emission_probs = {
        (i, o): emission_tolerance
        for i in hidden_states
        for o in observable_states
    }
    rowsum = {
        i: emission_tolerance * len(observable_states) for i in hidden_states
    }
    for sim in simulations:
        for i, curr in enumerate(sim.hidden):
            emission_probs[curr, sim.observed[i]] += 1
            rowsum[curr] += 1

    for i in hidden_states:
        if rowsum[i] == 0.0:
            if emission_prior:
                for o in observable_states:
                    emission_probs[i, o] = emission_prior.get((i, o), 0.0)
            else:
                # Uniform emission probabilities if we saw no emissions from
                # state i
                for o in observable_states:
                    emission_probs[i, o] = 1.0 / len(observable_states)
        else:
            # Normalize emission probabilities from state i
            for o in observable_states:
                emission_probs[i, o] /= rowsum[i]

    # import pprint
    # print()
    # pprint.pprint(start_probs)
    # print()
    # pprint.pprint(transition_probs)
    # print()
    # pprint.pprint(emission_probs)
    hmm = HMM()
    hmm.load_model(
        start_probs=start_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
    )
    return hmm
