import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.other_queries.sat_time_mvr import (  # noqa: E402
    sat_time_torch_mvr_chmm,
)

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


def make_reach_mvr(*, hidden_states, target_state, time_range=None, name=None):
    """Homogeneous MVR accepting once ``target_state`` has been seen; absorbing."""
    mediation_states = ["no", "yes"]

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini={h: ("yes" if h == target_state else "no") for h in hidden_states},
        upd={
            (m, h): ("yes" if m == "yes" or h == target_state else "no")
            for m in mediation_states
            for h in hidden_states
        },
        evl={"no": False, "yes": True},
        time_range=time_range,
        name=name,
    )


def mvr_window(mvr, T):
    """The window an MVR is enforced over, defaulting to the whole horizon."""
    time_range = getattr(mvr, "_time_range", None)

    return (0, T - 1) if time_range is None else tuple(time_range)


def mvr_first_sat_time(mvr, hidden, T):
    """Earliest time in ``[a, b]`` whose ``evl`` holds, or ``None``."""
    a, b = mvr_window(mvr, T)
    is_inhom = isinstance(mvr, InhomMVR)

    m = mvr.ini[hidden[a]]

    for t in range(a, b + 1):
        if t > a:
            if is_inhom:
                m = mvr.upd[t - a - 1][(m, hidden[t])]
            else:
                m = mvr.upd[(m, hidden[t])]

        if mvr.evl[t - a][m] if is_inhom else mvr.evl[m]:
            return t

    return None


def brute_force_sat_time(hmm, mvrs, target, observed, T):
    """First-satisfaction weights by enumeration, bucketed by first accept."""
    obs_map = as_obs_map(observed)
    others = [mvr for i, mvr in enumerate(mvrs) if i != target]
    a, b = mvr_window(mvrs[target], T)

    weights = np.zeros(b - a + 1)

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in others):
            continue

        hit = mvr_first_sat_time(mvrs[target], path, T)

        if hit is None:
            continue

        weights[hit - a] += math.exp(score_path(hmm, list(path), obs_map))

    return weights


def assert_matches_brute_force(hmm, mvrs, target, observed, T=None):
    if T is None:
        T = len(observed)

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected = brute_force_sat_time(hmm, mvrs, target, observed, T)

    if expected.sum() <= 0.0:
        with pytest.raises(InvalidInputError):
            sat_time_torch_mvr_chmm(model, observed, target=target, time_horizon=T)
        return None

    times, probs, log_weights = sat_time_torch_mvr_chmm(
        model, observed, target=target, time_horizon=T, return_log_weights=True
    )

    a, b = mvr_window(mvrs[target], T)

    assert times == list(range(a, b + 1))
    assert probs.shape == (b - a + 1,)

    # Both the shape of the distribution and, via log_weights, its absolute scale.
    assert probs.numpy() == pytest.approx(expected / expected.sum(), abs=1e-9)

    with np.errstate(divide="ignore"):
        assert log_weights.numpy() == pytest.approx(np.log(expected), abs=1e-9)

    return times, probs, log_weights


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
def test_sat_time_random_instances_match_brute_force(seed):
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

    def draw_window(min_span=0):
        if rng.random() < 0.4:
            return None
        start = int(rng.integers(0, max(1, T - min_span)))
        return [start, int(rng.integers(min(start + min_span, T - 1), T))]

    def draw_mvr(window, as_target):
        state = hidden_states[rng.integers(len(hidden_states))]
        span = (T - 1) if window is None else (window[1] - window[0])
        pick = rng.random()

        # forbid accepts at once and never again, so it makes a degenerate target.
        if as_target:
            pick = pick * 0.7 + 0.3

        if pick < 0.3:
            return make_forbid_mvr(
                hidden_states=hidden_states,
                forbidden_state=state,
                time_range=window,
            )
        if pick < 0.7:
            return make_reach_mvr(
                hidden_states=hidden_states,
                target_state=state,
                time_range=window,
            )
        if pick < 0.85:
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

    # Span at least two times where the horizon allows, so the target can spread.
    mvrs = [draw_mvr(draw_window(min_span=2), as_target=True)]
    target = 0

    for _ in range(int(rng.integers(0, 3))):
        mvrs.append(draw_mvr(draw_window(), as_target=False))

    # Move the target off position 0 to exercise the backward axis re-insertion.
    if len(mvrs) > 1:
        target = int(rng.integers(0, len(mvrs)))
        mvrs[0], mvrs[target] = mvrs[target], mvrs[0]

    assert_matches_brute_force(hmm, mvrs, target, observed, T)


# ===========================
# Target selection
# ===========================


def test_sat_time_target_by_index_name_and_negative_index_agree(hmm, observed):
    forbid = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="C")
    reach = make_reach_mvr(
        hidden_states=hmm.hidden_states, target_state="A", name="reach_A"
    )

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[forbid, reach])

    by_index = sat_time_torch_mvr_chmm(model, observed, target=1)
    by_negative = sat_time_torch_mvr_chmm(model, observed, target=-1)
    by_name = sat_time_torch_mvr_chmm(model, observed, target="reach_A")

    assert by_index[0] == by_negative[0] == by_name[0]
    assert by_negative[1].numpy() == pytest.approx(by_index[1].numpy())
    assert by_name[1].numpy() == pytest.approx(by_index[1].numpy())


# ===========================
# Error handling
# ===========================


def test_sat_time_raises_when_target_is_never_satisfied(hmm, observed):
    # No hidden state is named "Z", so the target can never accept.
    unreachable = make_reach_mvr(hidden_states=hmm.hidden_states, target_state="Z")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[unreachable])

    with pytest.raises(InvalidInputError, match="No feasible path"):
        sat_time_torch_mvr_chmm(model, observed, target=0)


@pytest.mark.parametrize(
    "constraints, target, match",
    [
        ([], 0, "at least one constraint"),
        (["reach"], 1, "out of range"),
        (["reach"], -2, "out of range"),
        (["reach"], "nope", "matches 0 constraints"),
        (["reach", "reach"], "reach_A", "matches 2 constraints"),
        (["reach"], 1.5, "integer index or a constraint name"),
        (["reach"], None, "integer index or a constraint name"),
    ],
)
def test_sat_time_rejects_bad_target(hmm, observed, constraints, target, match):
    mvrs = [
        make_reach_mvr(
            hidden_states=hmm.hidden_states, target_state="A", name="reach_A"
        )
        for _ in constraints
    ]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    with pytest.raises(InvalidInputError, match=match):
        sat_time_torch_mvr_chmm(model, observed, target=target)
