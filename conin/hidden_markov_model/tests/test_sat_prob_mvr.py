import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.other_queries.sat_prob_mvr import (  # noqa: E402
    sat_prob_torch_mvr_chmm,
)

from .test_sat_time_mvr import make_reach_mvr  # noqa: E402
from .test_viterbi_mvr import (  # noqa: E402
    as_obs_map,
    make_end_state_inhom_mvr,
    make_forbid_mvr,
    make_parity_mvr,
    make_random_hmm,
    mvr_accepts,
    score_path,
)

# ===========================
# Helpers
# ===========================


def brute_force_sat_prob(hmm, mvrs, target, observed, T):
    """``[satisfied, violated]`` weights, enumerated over paths feasible for the rest."""
    obs_map = as_obs_map(observed)
    others = [mvr for i, mvr in enumerate(mvrs) if i != target]

    weights = np.zeros(2)

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in others):
            continue

        branch = 0 if mvr_accepts(mvrs[target], path, T) else 1
        weights[branch] += math.exp(score_path(hmm, list(path), obs_map))

    return weights


def assert_matches_brute_force(hmm, mvrs, target, observed, T=None):
    if T is None:
        T = len(observed)

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected = brute_force_sat_prob(hmm, mvrs, target, observed, T)

    if expected.sum() <= 0.0:
        with pytest.raises(InvalidInputError):
            sat_prob_torch_mvr_chmm(model, observed, target=target, time_horizon=T)
        return None

    prob, log_weights = sat_prob_torch_mvr_chmm(
        model, observed, target=target, time_horizon=T, return_log_weights=True
    )

    # Both the normalized answer and, via log_weights, its absolute scale.
    assert prob == pytest.approx(expected[0] / expected.sum(), abs=1e-9)

    with np.errstate(divide="ignore"):
        assert log_weights.numpy() == pytest.approx(np.log(expected), abs=1e-9)

    return prob, log_weights


@pytest.fixture
def hmm():
    return make_random_hmm(
        hidden_states=["A", "B", "C"],
        observed_states=["o0", "o1"],
        seed=7,
    )


@pytest.fixture
def observed():
    return ["o0", "o1", "o1", "o0", "o1"]


# ===========================
# Correctness against brute force
# ===========================


@pytest.mark.parametrize("seed", range(40))
def test_sat_prob_random_instances_match_brute_force(seed):
    rng = np.random.default_rng(seed)

    hidden_states = [f"h{i}" for i in range(int(rng.integers(2, 4)))]
    observed_states = ["o0", "o1"]

    hmm = make_random_hmm(
        hidden_states=hidden_states,
        observed_states=observed_states,
        seed=seed,
    )

    T = int(rng.integers(1, 7))

    # Both observation forms, including a sparse map over a longer horizon.
    if rng.random() < 0.5:
        observed = [observed_states[rng.integers(2)] for _ in range(T)]
    else:
        times = rng.permutation(T)[: int(rng.integers(0, T + 1))]
        observed = {int(t): observed_states[rng.integers(2)] for t in times}

    def draw_window():
        if rng.random() < 0.4:
            return None
        start = int(rng.integers(0, T))
        return [start, int(rng.integers(start, T))]

    def draw_mvr(window):
        state = hidden_states[rng.integers(len(hidden_states))]
        span = (T - 1) if window is None else (window[1] - window[0])
        pick = rng.random()

        if pick < 0.3:
            return make_forbid_mvr(
                hidden_states=hidden_states,
                forbidden_state=state,
                time_range=window,
            )
        if pick < 0.55:
            return make_reach_mvr(
                hidden_states=hidden_states,
                target_state=state,
                time_range=window,
            )
        if pick < 0.8:
            return make_parity_mvr(
                hidden_states=hidden_states,
                target_state=state,
                time_range=window,
            )

        return make_end_state_inhom_mvr(
            hidden_states=hidden_states,
            target_state=state,
            time_horizon=span + int(rng.integers(0, 2)),
            time_range=window,
        )

    mvrs = [draw_mvr(draw_window()) for _ in range(int(rng.integers(1, 4)))]
    target = int(rng.integers(0, len(mvrs)))

    assert_matches_brute_force(hmm, mvrs, target, observed, T)


# ===========================
# Unsatisfiability
# ===========================


def test_sat_prob_answers_for_an_unsatisfiable_target_but_raises_for_the_rest(
    hmm, observed
):
    """Deliberately redundant with the sweep: pins the fork against sat_time, which
    raises for both. The sweep only covers it on the seeds that happen to land there."""
    # No hidden state is named "Z", so this one can never accept.
    unreachable = make_reach_mvr(hidden_states=hmm.hidden_states, target_state="Z")
    other = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[unreachable, other])

    prob, log_weights = sat_prob_torch_mvr_chmm(
        model, observed, target=0, return_log_weights=True
    )

    assert prob == 0.0
    assert log_weights[0] == -math.inf

    with pytest.raises(InvalidInputError, match="No feasible augmented path"):
        sat_prob_torch_mvr_chmm(model, observed, target=1)
