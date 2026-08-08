import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM
from conin.hidden_markov_model.hmm import HiddenMarkovModel
from conin.hidden_markov_model.mvr import HomMVR, InhomMVR

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.inference.viterbi_mvr import (  # noqa: E402
    viterbi_torch_mvr_chmm,
)

# ===========================
# Helpers
# ===========================


def make_random_hmm(*, hidden_states, observed_states, seed=0):
    rng = np.random.default_rng(seed)

    num_hidden = len(hidden_states)
    num_observed = len(observed_states)

    start_vec = rng.random(num_hidden)
    start_vec = start_vec / start_vec.sum()

    transition_mat = rng.random((num_hidden, num_hidden))
    transition_mat = transition_mat / transition_mat.sum(axis=1, keepdims=True)

    emission_mat = rng.random((num_hidden, num_observed))
    emission_mat = emission_mat / emission_mat.sum(axis=1, keepdims=True)

    hmm = HiddenMarkovModel()
    hmm.load_model(
        start_probs={h: start_vec[i] for i, h in enumerate(hidden_states)},
        transition_probs={
            (h1, h2): transition_mat[i, j]
            for i, h1 in enumerate(hidden_states)
            for j, h2 in enumerate(hidden_states)
        },
        emission_probs={
            (h, o): emission_mat[i, j]
            for i, h in enumerate(hidden_states)
            for j, o in enumerate(observed_states)
        },
        initialize=True,
    )

    return hmm


def make_forbid_mvr(*, hidden_states, forbidden_state, time_range=None):
    """Homogeneous MVR rejecting any path that visits ``forbidden_state``."""
    mediation_states = ["ok", "violated"]

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini={h: ("violated" if h == forbidden_state else "ok") for h in hidden_states},
        upd={
            (m, h): ("violated" if m == "violated" or h == forbidden_state else "ok")
            for m in mediation_states
            for h in hidden_states
        },
        evl={"ok": True, "violated": False},
        time_range=time_range,
    )


def make_parity_mvr(*, hidden_states, target_state, time_range=None):
    """Homogeneous MVR accepting iff ``target_state`` occurs an even number of times."""
    mediation_states = ["even", "odd"]
    flip = {"even": "odd", "odd": "even"}

    return HomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini={h: ("odd" if h == target_state else "even") for h in hidden_states},
        upd={
            (m, h): (flip[m] if h == target_state else m)
            for m in mediation_states
            for h in hidden_states
        },
        evl={"even": True, "odd": False},
        time_range=time_range,
    )


def make_end_state_inhom_mvr(
    *, hidden_states, target_state, time_horizon, time_range=None
):
    """Inhomogeneous MVR accepting iff the final state of its window is ``target_state``."""
    mediation_states = [[f"t{t}_no", f"t{t}_yes"] for t in range(time_horizon + 1)]

    return InhomMVR(
        hidden_states=hidden_states,
        mediation_states=mediation_states,
        ini={h: ("t0_yes" if h == target_state else "t0_no") for h in hidden_states},
        upd=[
            {
                (m, h): (f"t{t + 1}_yes" if h == target_state else f"t{t + 1}_no")
                for m in mediation_states[t]
                for h in hidden_states
            }
            for t in range(time_horizon)
        ],
        evl=[{f"t{t}_no": False, f"t{t}_yes": True} for t in range(time_horizon + 1)],
        time_range=time_range,
    )


def mvr_accepts(mvr, hidden, T):
    """Reference semantics: initialize at ``a``, run to ``b``, evaluate at ``b``."""
    time_range = getattr(mvr, "_time_range", None)
    a, b = (0, T - 1) if time_range is None else tuple(time_range)

    is_inhom = isinstance(mvr, InhomMVR)

    m = mvr.ini[hidden[a]]

    for t in range(a + 1, b + 1):
        if is_inhom:
            m = mvr.upd[t - a - 1][(m, hidden[t])]
        else:
            m = mvr.upd[(m, hidden[t])]

    return mvr.evl[b - a][m] if is_inhom else mvr.evl[m]


def brute_force(hmm, mvrs, observed):
    """Exhaustive search over hidden sequences; returns ``(path, log_prob)``."""
    T = len(observed)
    best_path, best_score = None, -math.inf

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in mvrs):
            continue

        score = hmm.log_probability(observed, list(path))

        if score > best_score:
            best_path, best_score = list(path), score

    return best_path, best_score


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


def assert_matches_brute_force(hmm, mvrs, observed):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected_path, expected_score = brute_force(hmm, mvrs, observed)

    path, augmented, score = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=True, return_score=True
    )

    assert len(path) == len(observed)
    assert len(augmented) == len(observed)

    # The decoded path must be feasible and optimal, though not necessarily the
    # same optimal path brute force found when several are tied.
    assert all(mvr_accepts(mvr, path, len(observed)) for mvr in mvrs)
    assert hmm.log_probability(observed, path) == pytest.approx(
        expected_score, abs=1e-4
    )
    assert score == pytest.approx(expected_score, abs=1e-4)

    return path, augmented, score


# ===========================
# Correctness against brute force
# ===========================


def test_viterbi_no_constraints_matches_brute_force(hmm, observed):
    assert_matches_brute_force(hmm, [], observed)


def test_viterbi_hom_mvr_full_range(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    assert "B" not in path


def test_viterbi_hom_mvr_windowed(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[1, 3],
    )
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    # The constraint binds only inside its window.
    assert "B" not in path[1:4]


def test_viterbi_hom_mvr_single_time_window(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[2, 2],
    )
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    assert path[2] != "B"


def test_viterbi_inhom_mvr_full_range(hmm, observed):
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="A",
        time_horizon=len(observed) - 1,
    )
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    assert path[-1] == "A"


def test_viterbi_inhom_mvr_windowed(hmm, observed):
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="A",
        time_horizon=2,
        time_range=[1, 3],
    )
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    assert path[3] == "A"


def test_viterbi_inhom_mvr_horizon_longer_than_window(hmm, observed):
    # time_horizon 4 is longer than the width-2 window; only the first 2
    # transitions and the time-2 evaluation are used.
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="A",
        time_horizon=4,
        time_range=[1, 3],
    )
    assert_matches_brute_force(hmm, [mvr], observed)


def test_viterbi_disjoint_windows(hmm, observed):
    mvrs = [
        make_forbid_mvr(
            hidden_states=hmm.hidden_states,
            forbidden_state="B",
            time_range=[0, 1],
        ),
        make_forbid_mvr(
            hidden_states=hmm.hidden_states,
            forbidden_state="C",
            time_range=[3, 4],
        ),
    ]
    path, _, _ = assert_matches_brute_force(hmm, mvrs, observed)

    assert "B" not in path[0:2]
    assert "C" not in path[3:5]


def test_viterbi_overlapping_windows_mixed_types(hmm, observed):
    mvrs = [
        make_end_state_inhom_mvr(
            hidden_states=hmm.hidden_states,
            target_state="C",
            time_horizon=2,
            time_range=[0, 2],
        ),
        make_parity_mvr(
            hidden_states=hmm.hidden_states,
            target_state="A",
            time_range=[2, 4],
        ),
    ]
    assert_matches_brute_force(hmm, mvrs, observed)


def test_viterbi_three_overlapping_windows(hmm, observed):
    mvrs = [
        make_forbid_mvr(
            hidden_states=hmm.hidden_states,
            forbidden_state="B",
            time_range=[0, 2],
        ),
        make_parity_mvr(
            hidden_states=hmm.hidden_states,
            target_state="A",
            time_range=[1, 4],
        ),
        make_forbid_mvr(
            hidden_states=hmm.hidden_states,
            forbidden_state="C",
            time_range=[2, 3],
        ),
    ]
    assert_matches_brute_force(hmm, mvrs, observed)


def test_viterbi_single_time_step(hmm):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    path, _, _ = assert_matches_brute_force(hmm, [mvr], ["o1"])

    assert len(path) == 1
    assert path[0] != "B"


def test_viterbi_mvr_hidden_state_ordering_is_permuted(hmm, observed):
    """The MVR's hidden-state ordering need not match the HMM's."""
    permuted = list(reversed(hmm.hidden_states))
    assert permuted != hmm.hidden_states

    mvr = make_forbid_mvr(hidden_states=permuted, forbidden_state="B")
    path, _, _ = assert_matches_brute_force(hmm, [mvr], observed)

    assert "B" not in path


# ===========================
# Randomized cross-check
# ===========================


@pytest.mark.parametrize("seed", range(25))
def test_viterbi_random_instances_match_brute_force(seed):
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

    for _ in range(int(rng.integers(0, 3))):
        time_range = None
        if rng.random() < 0.6:
            a = int(rng.integers(0, T))
            time_range = [a, int(rng.integers(a, T))]

        if rng.random() < 0.5:
            mvrs.append(
                make_forbid_mvr(
                    hidden_states=hidden_states,
                    forbidden_state=hidden_states[rng.integers(len(hidden_states))],
                    time_range=time_range,
                )
            )
        else:
            span = (T - 1) if time_range is None else (time_range[1] - time_range[0])
            mvrs.append(
                make_end_state_inhom_mvr(
                    hidden_states=hidden_states,
                    target_state=hidden_states[rng.integers(len(hidden_states))],
                    time_horizon=span + int(rng.integers(0, 2)),
                    time_range=time_range,
                )
            )

    expected_path, expected_score = brute_force(hmm, mvrs, observed)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    if expected_path is None:
        with pytest.raises(InvalidInputError):
            viterbi_torch_mvr_chmm(model, observed, return_augmented=False)
        return

    path, score = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=False, return_score=True
    )

    assert all(mvr_accepts(mvr, path, T) for mvr in mvrs)
    assert hmm.log_probability(observed, path) == pytest.approx(
        expected_score, abs=1e-4
    )
    assert score == pytest.approx(expected_score, abs=1e-4)


# ===========================
# Score, options, and error handling
# ===========================


def test_viterbi_score_is_path_log_probability(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    path, score = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=False, return_score=True
    )

    assert score == pytest.approx(hmm.log_probability(observed, path), abs=1e-4)


def test_viterbi_return_shapes(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    path = viterbi_torch_mvr_chmm(model, observed, return_augmented=False)
    assert isinstance(path, list)

    path, score = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=False, return_score=True
    )
    assert isinstance(score, float)

    path, augmented = viterbi_torch_mvr_chmm(model, observed)
    assert len(augmented) == len(observed)

    path, augmented, score = viterbi_torch_mvr_chmm(model, observed, return_score=True)
    assert isinstance(score, float)


def test_viterbi_augmented_path_reports_mediation_states(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="B")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    path, augmented = viterbi_torch_mvr_chmm(model, observed)

    for t, entry in enumerate(augmented):
        assert hmm.hidden_to_external[entry["hidden_index"]] == path[t]
        # The MVR is active throughout and must never enter "violated".
        assert entry["mvr_states"][0] == "ok"


def test_viterbi_augmented_path_omits_inactive_mvrs(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[1, 3],
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    _, augmented = viterbi_torch_mvr_chmm(model, observed)

    for t, entry in enumerate(augmented):
        if 1 <= t <= 3:
            assert 0 in entry["mvr_states"]
        else:
            assert entry["mvr_states"] == {}


def test_viterbi_rejects_empty_observed(hmm):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match="nonempty"):
        viterbi_torch_mvr_chmm(model, [])


def test_viterbi_rejects_unknown_observed_state(hmm):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match="Unknown observed state"):
        viterbi_torch_mvr_chmm(model, ["o0", "not_an_observation"])


def test_viterbi_raises_on_infeasible_constraints(hmm, observed):
    mvrs = [
        make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state=h)
        for h in hmm.hidden_states
    ]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    with pytest.raises(InvalidInputError, match="No feasible augmented path"):
        viterbi_torch_mvr_chmm(model, observed, return_augmented=False)


def test_viterbi_rejects_time_range_beyond_horizon(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[2, len(observed) + 3],
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    with pytest.raises(InvalidInputError, match="exceeds"):
        viterbi_torch_mvr_chmm(model, observed, return_augmented=False)


def test_viterbi_rejects_inhom_mvr_with_too_short_horizon(hmm, observed):
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="A",
        time_horizon=1,
        time_range=[0, 1],
    )
    # Widen the window past what the MVR's own horizon can support.
    mvr._time_range = [0, 4]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    with pytest.raises(InvalidInputError, match="time_horizon is too short"):
        viterbi_torch_mvr_chmm(model, observed, return_augmented=False)


# ===========================
# Horizon and sparse observations
# ===========================


def as_obs_map(observed):
    """Normalize either observation form to a ``{time: label}`` map."""
    if isinstance(observed, dict):
        return dict(observed)
    return dict(enumerate(observed))


def score_path(hmm, path, obs_map):
    """Log probability of a hidden path, scoring only the times that are observed."""
    repn = hmm.repn
    idx = [hmm.hidden_to_internal[h] for h in path]

    score = math.log(repn.start_vec[idx[0]])

    for t in range(1, len(idx)):
        score += math.log(repn.transition_mat[idx[t - 1]][idx[t]])

    for t, o in obs_map.items():
        score += math.log(repn.emission_mat[idx[t]][hmm.observed_to_internal[o]])

    return score


def brute_force_general(hmm, mvrs, observed, T):
    """Exhaustive search over a horizon, scoring only the times that are observed."""
    obs_map = as_obs_map(observed)

    best_path, best_score = None, -math.inf

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in mvrs):
            continue

        score = score_path(hmm, path, obs_map)

        if score > best_score:
            best_path, best_score = list(path), score

    return best_path, best_score


def test_brute_force_general_agrees_with_dense_reference(hmm, observed):
    # Guards the new reference implementation against the established one.
    path, score = brute_force_general(hmm, [], observed, len(observed))
    expected_path, expected_score = brute_force(hmm, [], observed)

    assert path == expected_path
    assert score == pytest.approx(expected_score, abs=1e-9)


def test_viterbi_dict_observations_match_dense_list(hmm, observed):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    dense = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=False, return_score=True
    )
    sparse = viterbi_torch_mvr_chmm(
        model,
        dict(enumerate(observed)),
        time_horizon=len(observed),
        return_augmented=False,
        return_score=True,
    )

    assert dense == sparse


def test_viterbi_horizon_longer_than_observations(hmm, observed):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    T = len(observed) + 2

    path, score = viterbi_torch_mvr_chmm(
        model, observed, time_horizon=T, return_augmented=False, return_score=True
    )
    expected_path, expected_score = brute_force_general(hmm, [], observed, T)

    assert len(path) == T
    assert path == expected_path
    assert score == pytest.approx(expected_score, abs=1e-4)


def test_viterbi_sparse_observations_match_brute_force(hmm):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    sparse = {0: "o0", 3: "o1", 4: "o1"}

    path, score = viterbi_torch_mvr_chmm(
        model, sparse, time_horizon=5, return_augmented=False, return_score=True
    )
    expected_path, expected_score = brute_force_general(hmm, [], sparse, 5)

    # An absent observation drops the emission factor, never the time step.
    assert len(path) == 5
    assert path == expected_path
    assert score == pytest.approx(expected_score, abs=1e-4)


def test_viterbi_no_observations_is_prior_map(hmm):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    from_map = viterbi_torch_mvr_chmm(
        model, {}, time_horizon=4, return_augmented=False, return_score=True
    )
    from_list = viterbi_torch_mvr_chmm(
        model, [], time_horizon=4, return_augmented=False, return_score=True
    )
    expected_path, expected_score = brute_force_general(hmm, [], {}, 4)

    assert from_map == from_list
    assert from_map[0] == expected_path
    assert from_map[1] == pytest.approx(expected_score, abs=1e-4)


def test_viterbi_unwindowed_mvr_covers_the_extended_horizon(hmm, observed):
    # A time_range-less MVR defaults to [0, T-1], so extending the horizon
    # extends enforcement past the observed span.
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])
    T = len(observed) + 3

    path = viterbi_torch_mvr_chmm(
        model, observed, time_horizon=T, return_augmented=False
    )

    assert len(path) == T
    assert "A" not in path


@pytest.mark.parametrize("seed", range(6))
def test_viterbi_random_sparse_instances_match_brute_force(seed):
    rng = np.random.default_rng(1000 + seed)

    hmm = make_random_hmm(
        hidden_states=["A", "B", "C"],
        observed_states=["o0", "o1"],
        seed=seed,
    )

    T = int(rng.integers(2, 6))
    times = rng.permutation(T)[: int(rng.integers(0, T + 1))]
    sparse = {int(t): str(rng.choice(["o0", "o1"])) for t in times}

    mvrs = []
    if rng.random() < 0.6:
        end = int(rng.integers(0, T))
        start = int(rng.integers(0, end + 1))
        mvrs.append(
            make_forbid_mvr(
                hidden_states=hmm.hidden_states,
                forbidden_state=str(rng.choice(hmm.hidden_states)),
                time_range=[start, end],
            )
        )

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
    expected_path, expected_score = brute_force_general(hmm, mvrs, sparse, T)

    if expected_path is None:
        with pytest.raises(InvalidInputError):
            viterbi_torch_mvr_chmm(
                model, sparse, time_horizon=T, return_augmented=False
            )
        return

    path, score = viterbi_torch_mvr_chmm(
        model, sparse, time_horizon=T, return_augmented=False, return_score=True
    )

    # Compare on score rather than path: distinct paths can tie exactly.
    assert score == pytest.approx(expected_score, abs=1e-4)
    assert len(path) == T
    assert all(mvr_accepts(mvr, path, T) for mvr in mvrs)
    assert score_path(hmm, path, as_obs_map(sparse)) == pytest.approx(score, abs=1e-4)


@pytest.mark.parametrize(
    "observed_arg, horizon, message",
    [
        ({0: "o0"}, None, "time_horizon is required"),
        (["o0", "o1"], 1, "shorter than the observed"),
        ({7: "o0"}, 3, "outside the horizon"),
        ({}, 0, "positive integer"),
    ],
)
def test_viterbi_rejects_bad_horizon_or_observations(
    hmm, observed_arg, horizon, message
):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match=message):
        viterbi_torch_mvr_chmm(model, observed_arg, time_horizon=horizon)


def test_viterbi_rejects_inhom_mvr_too_short_for_extended_horizon(hmm, observed):
    # Fits len(observed) == 5, but not the extended horizon.
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="A",
        time_horizon=len(observed) - 1,
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    viterbi_torch_mvr_chmm(model, observed, return_augmented=False)

    with pytest.raises(InvalidInputError, match="time_horizon is too short"):
        viterbi_torch_mvr_chmm(
            model, observed, time_horizon=len(observed) + 1, return_augmented=False
        )
