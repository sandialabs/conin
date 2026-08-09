import collections
import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.sampling.ffbs_mvr import (  # noqa: E402
    ffbs_torch_mvr_chmm,
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

# Draws per sweep case.
NUM_SAMPLES = 20000

# ===========================
# Helpers
# ===========================


def brute_force_path_posterior(hmm, mvrs, observed, T):
    """Exact posterior over feasible hidden paths, by enumeration."""
    obs_map = as_obs_map(observed)

    weights = {
        path: math.exp(score_path(hmm, list(path), obs_map))
        for path in itertools.product(hmm.hidden_states, repeat=T)
        if all(mvr_accepts(mvr, path, T) for mvr in mvrs)
    }

    total = sum(weights.values())

    return {path: w / total for path, w in weights.items()}


def null_tv_mean(probs):
    """Mean total variation between a correct sample and its own distribution."""
    return 0.5 * sum(
        math.sqrt(2 * p * (1 - p) / (math.pi * NUM_SAMPLES)) for p in probs
    )


def assert_matches_brute_force(hmm, mvrs, observed, T=None, seed=0):
    """Check drawn paths against enumeration: feasibility exactly, shape by TV."""
    if T is None:
        T = len(observed)

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected = brute_force_path_posterior(hmm, mvrs, observed, T)

    if not expected:
        with pytest.raises(InvalidInputError):
            ffbs_torch_mvr_chmm(model, observed, time_horizon=T)
        return None

    paths = ffbs_torch_mvr_chmm(
        model,
        observed,
        num_samples=NUM_SAMPLES,
        time_horizon=T,
        generator=torch.Generator().manual_seed(seed),
    )

    assert all(len(path) == T for path in paths)

    counts = collections.Counter(tuple(path) for path in paths)

    # An infeasible draw is a bug however the distribution looks.
    assert set(counts) <= set(expected)

    total_variation = 0.5 * sum(
        abs(counts.get(path, 0) / NUM_SAMPLES - p) for path, p in expected.items()
    )

    assert total_variation <= 5 * null_tv_mean(expected.values())

    return paths


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


@pytest.mark.parametrize("seed", range(25))
def test_ffbs_random_instances_match_brute_force(seed):
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

    mvrs = []

    # At least one: the constraint machinery is what this is testing.
    for _ in range(int(rng.integers(1, 4))):
        time_range = None
        if rng.random() < 0.6:
            a = int(rng.integers(0, T))
            time_range = [a, int(rng.integers(a, T))]

        shape = rng.integers(3)
        state = hidden_states[rng.integers(len(hidden_states))]

        if shape == 0:
            mvrs.append(
                make_forbid_mvr(
                    hidden_states=hidden_states,
                    forbidden_state=state,
                    time_range=time_range,
                )
            )
        elif shape == 1:
            mvrs.append(
                make_parity_mvr(
                    hidden_states=hidden_states,
                    target_state=state,
                    time_range=time_range,
                )
            )
        else:
            span = (T - 1) if time_range is None else (time_range[1] - time_range[0])
            mvrs.append(
                make_end_state_inhom_mvr(
                    hidden_states=hidden_states,
                    target_state=state,
                    time_horizon=span + int(rng.integers(0, 2)),
                    time_range=time_range,
                )
            )

    # Half the cases drop to a sparse map, leaving some times unobserved.
    if rng.random() < 0.5:
        observed = {t: o for t, o in enumerate(observed) if rng.random() < 0.7}

    assert_matches_brute_force(hmm, mvrs, observed, T=T, seed=seed)


# ===========================
# Options and error handling
# ===========================


def test_ffbs_is_reproducible_under_a_seeded_generator(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    def draw(seed):
        return ffbs_torch_mvr_chmm(
            model,
            observed,
            num_samples=32,
            generator=torch.Generator().manual_seed(seed),
            return_indices=True,
        )[1]

    assert torch.equal(draw(4), draw(4))
    assert not torch.equal(draw(4), draw(5))


def test_ffbs_return_shapes(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states, forbidden_state="B", time_range=[1, 3]
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    paths, indices, augmented = ffbs_torch_mvr_chmm(
        model,
        observed,
        num_samples=6,
        generator=torch.Generator().manual_seed(0),
        return_indices=True,
        return_augmented=True,
    )

    T = len(observed)

    assert len(paths) == len(augmented) == 6
    assert indices.shape == (6, T)

    for n in range(6):
        assert [hmm.hidden_to_external[i] for i in indices[n].tolist()] == paths[n]
        assert [entry["hidden_index"] for entry in augmented[n]] == indices[n].tolist()

        # The MVR is windowed, so it reports a mediation state only inside [1, 3].
        for t, entry in enumerate(augmented[n]):
            assert entry["mvr_states"] == ({0: "ok"} if 1 <= t <= 3 else {})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"num_samples": True}, "num_samples"),
        ({"time_horizon": 2}, "shorter than"),
    ],
)
def test_ffbs_rejects_bad_arguments(hmm, observed, kwargs, match):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match=match):
        ffbs_torch_mvr_chmm(model, observed, **kwargs)
