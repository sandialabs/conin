import collections
import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.mvr_operators import mvr_sattime  # noqa: E402
from conin.hidden_markov_model.sampling.stopped_sampling_mvr import (  # noqa: E402
    stopped_sampling_torch_mvr_chmm,
)

from .test_ffbs_mvr import NUM_SAMPLES, null_tv_mean  # noqa: E402
from .test_sat_time_mvr import (  # noqa: E402
    brute_force_sat_time,
    make_reach_mvr,
    mvr_first_sat_time,
    mvr_window,
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


def brute_force_stopped_joint(hmm, mvrs, target, observed, T):
    """Exact joint over ``(stopping time, prefix)`` by enumeration."""
    obs_map = as_obs_map(observed)
    others = [mvr for i, mvr in enumerate(mvrs) if i != target]

    weights = collections.defaultdict(float)

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in others):
            continue

        tau = mvr_first_sat_time(mvrs[target], path, T)

        if tau is None:
            continue

        weights[(tau, path[: tau + 1])] += math.exp(
            score_path(hmm, list(path), obs_map)
        )

    total = sum(weights.values())

    return {key: w / total for key, w in weights.items()}


def assert_matches_brute_force(hmm, mvrs, target, observed, T=None, seed=0):
    """Check the sampled ``(stopping time, prefix)`` pairs against enumeration."""
    if T is None:
        T = len(observed)

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected = brute_force_stopped_joint(hmm, mvrs, target, observed, T)

    if not expected:
        with pytest.raises(InvalidInputError):
            stopped_sampling_torch_mvr_chmm(
                model, observed, target=target, time_horizon=T
            )
        return None

    paths, times = stopped_sampling_torch_mvr_chmm(
        model,
        observed,
        target=target,
        num_samples=NUM_SAMPLES,
        time_horizon=T,
        generator=torch.Generator().manual_seed(seed),
        return_times=True,
    )

    # The returned prefix must be the one its reported stopping time describes.
    assert [len(path) - 1 for path in paths] == times.tolist()

    counts = collections.Counter(
        (len(path) - 1, tuple(path)) for path in paths
    )

    assert set(counts) <= set(expected)

    total_variation = 0.5 * sum(
        abs(counts.get(key, 0) / NUM_SAMPLES - p) for key, p in expected.items()
    )

    assert total_variation <= 5 * null_tv_mean(expected.values())

    return paths, times


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


@pytest.fixture
def target():
    return make_reach_mvr(
        hidden_states=["A", "B", "C"], target_state="C", name="hit"
    )


# ===========================
# Correctness against brute force
# ===========================


@pytest.mark.parametrize("seed", range(20))
def test_stopped_sampling_random_instances_match_brute_force(seed):
    rng = np.random.default_rng(seed)

    hidden_states = [f"h{i}" for i in range(int(rng.integers(2, 4)))]
    observed_states = ["o0", "o1"]

    hmm = make_random_hmm(
        hidden_states=hidden_states,
        observed_states=observed_states,
        seed=seed,
    )

    T = int(rng.integers(1, 6))
    observed = [observed_states[rng.integers(2)] for _ in range(T)]

    def draw_window():
        if rng.random() < 0.5:
            return None
        a = int(rng.integers(0, T))
        return [a, int(rng.integers(a, T))]

    # The target is neither prefix-free shape, so both exercise the substitution.
    window = draw_window()
    state = hidden_states[rng.integers(len(hidden_states))]

    if rng.random() < 0.5:
        mvrs = [
            make_reach_mvr(
                hidden_states=hidden_states,
                target_state=state,
                time_range=window,
            )
        ]
    else:
        span = (T - 1) if window is None else (window[1] - window[0])
        mvrs = [
            make_end_state_inhom_mvr(
                hidden_states=hidden_states,
                target_state=state,
                time_horizon=span + int(rng.integers(0, 2)),
                time_range=window,
            )
        ]

    for _ in range(int(rng.integers(0, 3))):
        other = draw_window()
        state = hidden_states[rng.integers(len(hidden_states))]

        if rng.random() < 0.5:
            mvrs.append(
                make_forbid_mvr(
                    hidden_states=hidden_states,
                    forbidden_state=state,
                    time_range=other,
                )
            )
        else:
            mvrs.append(
                make_parity_mvr(
                    hidden_states=hidden_states,
                    target_state=state,
                    time_range=other,
                )
            )

    if rng.random() < 0.5:
        observed = {t: o for t, o in enumerate(observed) if rng.random() < 0.7}

    assert_matches_brute_force(hmm, mvrs, 0, observed, T=T, seed=seed)


# ===========================
# The prefix-free substitution
# ===========================


def test_prefix_free_substitution_is_transparent(hmm, observed, target, capsys):
    already = mvr_sattime(target)
    already._time_range = target._time_range
    already._name = target._name

    def draw(mvr):
        model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

        return stopped_sampling_torch_mvr_chmm(
            model,
            observed,
            target=0,
            num_samples=64,
            generator=torch.Generator().manual_seed(3),
        )

    assert draw(target) == draw(already)

    # An already prefix-free target must not reach mvr_sattime, which prints.
    assert capsys.readouterr().out == ""


def test_stopped_sampling_leaves_the_caller_model_untouched(hmm, observed, target):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[target])

    stored = model.constraints[0]
    constraints = model.constraints

    stopped_sampling_torch_mvr_chmm(
        model,
        observed,
        target="hit",
        num_samples=8,
        generator=torch.Generator().manual_seed(0),
    )

    assert model.constraints is constraints
    assert model.constraints[0] is stored
    assert stored.prefix is False
    assert stored._time_range is None


# ===========================
# min_length
# ===========================


@pytest.mark.parametrize("min_length", [1, 2, 4, 5])
def test_min_length_restricts_the_stopping_time(hmm, observed, target, min_length):
    other = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    mvrs = [target, other]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    T = len(observed)
    a, _ = mvr_window(target, T)

    paths, times = stopped_sampling_torch_mvr_chmm(
        model,
        observed,
        target=0,
        num_samples=NUM_SAMPLES,
        min_length=min_length,
        time_horizon=T,
        generator=torch.Generator().manual_seed(1),
        return_times=True,
    )

    assert min(len(path) for path in paths) >= min_length

    # The surviving stopping times keep their relative satisfaction-time weights.
    expected = brute_force_sat_time(hmm, mvrs, 0, observed, T)
    expected[: max(0, min_length - 1 - a)] = 0.0
    expected = expected / expected.sum()

    counts = collections.Counter(times.tolist())
    empirical = np.array(
        [counts.get(t, 0) / NUM_SAMPLES for t in range(a, T)]
    )

    assert 0.5 * np.abs(empirical - expected).sum() <= 5 * null_tv_mean(expected)


def test_min_length_below_the_window_is_vacuous(hmm, observed, target):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[target])

    def draw(min_length):
        return stopped_sampling_torch_mvr_chmm(
            model,
            observed,
            target=0,
            num_samples=48,
            min_length=min_length,
            generator=torch.Generator().manual_seed(2),
        )

    assert draw(1) == draw(None)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_length": 0}, "min_length"),
        ({"min_length": 1.5}, "min_length"),
        ({"min_length": True}, "min_length"),
        ({"min_length": 99}, "leaves no feasible stopping time"),
        ({"num_samples": 0}, "num_samples"),
        ({"target": "nope"}, "matches 0 constraints"),
    ],
)
def test_stopped_sampling_rejects_bad_arguments(hmm, observed, target, kwargs, match):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[target])

    kwargs = {"target": 0, **kwargs}

    with pytest.raises(InvalidInputError, match=match):
        stopped_sampling_torch_mvr_chmm(model, observed, **kwargs)
