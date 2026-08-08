import contextlib
import itertools
import math

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.inference.marginal_map_mvr import (  # noqa: E402
    marginal_map_torch_mvr_chmm,
)
from conin.hidden_markov_model.inference.viterbi_mvr import (  # noqa: E402
    viterbi_torch_mvr_chmm,
)
from .test_viterbi_mvr import (  # noqa: E402
    as_obs_map,
    make_end_state_inhom_mvr,
    make_forbid_mvr,
    make_random_hmm,
    mvr_accepts,
    score_path,
)


# ===========================
# Reference implementation
# ===========================


def brute_force_marginal_map(hmm, mvrs, observed, T, query_times):
    """
    Exhaustive marginal MAP: group feasible paths by their query-time states,
    sum the probability within each group, and take the heaviest group.
    """
    obs_map = as_obs_map(observed)
    groups = {}

    for path in itertools.product(hmm.hidden_states, repeat=T):
        if not all(mvr_accepts(mvr, path, T) for mvr in mvrs):
            continue

        key = tuple(path[t] for t in query_times)
        groups[key] = groups.get(key, 0.0) + math.exp(score_path(hmm, path, obs_map))

    if not groups:
        return None, -math.inf

    best = max(groups, key=groups.get)

    return list(best), math.log(groups[best])


def assert_matches_brute_force(hmm, mvrs, observed, T, query_times):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    expected_path, expected_score = brute_force_marginal_map(
        hmm, mvrs, observed, T, query_times
    )

    path, score = marginal_map_torch_mvr_chmm(
        model,
        observed,
        time_horizon=T,
        query_times=query_times,
        return_augmented=False,
        return_score=True,
    )

    assert path == expected_path
    assert score == pytest.approx(expected_score, abs=1e-4)


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
# Agreement with brute force
# ===========================


@pytest.mark.parametrize(
    "query_times",
    # empty prefix, empty suffix, all three segments, adjacent + gap mix.
    [[0], [4], [1, 3], [0, 1, 4]],
)
def test_marginal_map_matches_brute_force(hmm, observed, query_times):
    assert_matches_brute_force(hmm, [], observed, len(observed), query_times)


@pytest.mark.parametrize("query_times", [[0, 4], [0, 2, 4]])
def test_marginal_map_with_constraint_matches_brute_force(hmm, observed, query_times):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    assert_matches_brute_force(hmm, [mvr], observed, len(observed), query_times)


def test_marginal_map_with_inhom_constraint(hmm, observed):
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="C",
        time_horizon=len(observed) - 1,
    )
    assert_matches_brute_force(hmm, [mvr], observed, len(observed), [0, 2, 4])


def test_marginal_map_mvr_window_entirely_inside_a_gap(hmm, observed):
    # Both ini and evl land on summed-out times, so the whole automaton runs
    # and is judged inside the transfer operator.
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="A",
        time_range=[1, 2],
    )
    assert_matches_brute_force(hmm, [mvr], observed, len(observed), [0, 3])


def test_marginal_map_mvr_window_inside_the_prefix(hmm, observed):
    # The window closes before the first query time, so evl is applied by the
    # forward prefix pass.
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[0, 1],
    )
    assert_matches_brute_force(hmm, [mvr], observed, len(observed), [3, 4])


def test_marginal_map_mvr_window_inside_the_suffix(hmm, observed):
    # The window opens after the last query time, so the whole automaton is
    # handled by the backward suffix pass.
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[3, 4],
    )
    assert_matches_brute_force(hmm, [mvr], observed, len(observed), [0, 1])


def test_marginal_map_two_constraints_over_a_gap(hmm, observed):
    mvrs = [
        make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A"),
        make_forbid_mvr(
            hidden_states=hmm.hidden_states,
            forbidden_state="C",
            time_range=[0, 2],
        ),
    ]
    assert_matches_brute_force(hmm, mvrs, observed, len(observed), [0, 3])


@pytest.mark.parametrize("seed", range(6))
def test_marginal_map_random_instances_match_brute_force(seed):
    rng = np.random.default_rng(2000 + seed)

    hmm = make_random_hmm(
        hidden_states=["A", "B", "C"],
        observed_states=["o0", "o1"],
        seed=seed,
    )

    T = int(rng.integers(2, 6))
    observed = [str(rng.choice(["o0", "o1"])) for _ in range(T)]

    num_query = int(rng.integers(1, T + 1))
    query_times = sorted(int(t) for t in rng.permutation(T)[:num_query])

    mvrs = []
    if rng.random() < 0.7:
        end = int(rng.integers(0, T))
        start = int(rng.integers(0, end + 1))
        mvrs.append(
            make_forbid_mvr(
                hidden_states=hmm.hidden_states,
                forbidden_state=str(rng.choice(hmm.hidden_states)),
                time_range=[start, end],
            )
        )

    expected_path, _ = brute_force_marginal_map(hmm, mvrs, observed, T, query_times)

    if expected_path is None:
        model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)
        with pytest.raises(InvalidInputError):
            marginal_map_torch_mvr_chmm(
                model, observed, query_times=query_times, return_augmented=False
            )
        return

    with pytest.warns(UserWarning) if num_query == T else contextlib.nullcontext():
        assert_matches_brute_force(hmm, mvrs, observed, T, query_times)


# ===========================
# Relationship to Viterbi
# ===========================


def test_marginal_map_over_all_times_equals_viterbi(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    expected = viterbi_torch_mvr_chmm(
        model, observed, return_augmented=False, return_score=True
    )

    with pytest.warns(UserWarning, match="reduces to Viterbi"):
        actual = marginal_map_torch_mvr_chmm(
            model, observed, return_augmented=False, return_score=True
        )

    assert actual[0] == expected[0]
    assert actual[1] == pytest.approx(expected[1], abs=1e-4)


def test_marginal_map_differs_from_restricting_viterbi(hmm, observed):
    # Maximizing a marginal is not marginalizing a maximum. This fixture is a
    # concrete witness, so a refactor that silently turns one into the other
    # cannot pass unnoticed.
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    query_times = [0, 2, 4]

    viterbi_path = viterbi_torch_mvr_chmm(model, observed, return_augmented=False)
    restricted = [viterbi_path[t] for t in query_times]

    marginal_path = marginal_map_torch_mvr_chmm(
        model, observed, query_times=query_times, return_augmented=False
    )

    assert marginal_path != restricted
    assert marginal_path == brute_force_marginal_map(
        hmm, [], observed, len(observed), query_times
    )[0]


# ===========================
# Horizon and sparse observations
# ===========================


def test_marginal_map_horizon_longer_than_observations(hmm, observed):
    assert_matches_brute_force(hmm, [], observed, len(observed) + 2, [0, 3, 6])


def test_marginal_map_sparse_observations(hmm):
    sparse = {0: "o0", 3: "o1"}
    assert_matches_brute_force(hmm, [], sparse, 5, [1, 4])


# ===========================
# Gap fast path
# ===========================


def test_bare_gap_matches_step_by_step_propagation(hmm):
    from conin.hidden_markov_model.inference import marginal_map_mvr as mod

    sparse = {0: "o0", 5: "o1"}
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    fast = marginal_map_torch_mvr_chmm(
        model, sparse, time_horizon=6, query_times=[0, 5], return_score=True
    )

    # Disable the transition-power shortcut and redo it the general way.
    original = mod._gap_is_bare
    mod._gap_is_bare = lambda *args, **kwargs: False
    try:
        general = marginal_map_torch_mvr_chmm(
            model, sparse, time_horizon=6, query_times=[0, 5], return_score=True
        )
    finally:
        mod._gap_is_bare = original

    assert fast[0] == general[0]
    assert fast[2] == pytest.approx(general[2], abs=1e-4)


# ===========================
# Output shape and validation
# ===========================


def test_marginal_map_augmented_path_reports_query_times(hmm, observed):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states,
        forbidden_state="B",
        time_range=[2, 4],
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])
    query_times = [0, 2, 4]

    path, augmented = marginal_map_torch_mvr_chmm(
        model, observed, query_times=query_times
    )

    assert len(path) == len(query_times)
    assert [entry["time"] for entry in augmented] == query_times

    for entry in augmented:
        assert hmm.hidden_to_external[entry["hidden_index"]] == path[
            query_times.index(entry["time"])
        ]
        # The MVR is inactive at time 0 and active from time 2 on.
        if entry["time"] == 0:
            assert entry["mvr_states"] == {}
        else:
            assert entry["mvr_states"][0] == "ok"


def test_marginal_map_unsorted_and_duplicate_query_times(hmm, observed):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    messy = marginal_map_torch_mvr_chmm(
        model, observed, query_times=[4, 0, 2, 0], return_augmented=False
    )
    tidy = marginal_map_torch_mvr_chmm(
        model, observed, query_times=[0, 2, 4], return_augmented=False
    )

    assert messy == tidy


@pytest.mark.parametrize(
    "query_times, message",
    [
        ([0, 99], "outside the horizon"),
        ([], "at least one time"),
        ([0, "two"], "must be an integer"),
    ],
)
def test_marginal_map_rejects_bad_query_times(hmm, observed, query_times, message):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match=message):
        marginal_map_torch_mvr_chmm(model, observed, query_times=query_times)


def test_marginal_map_raises_on_infeasible_constraints(hmm, observed):
    mvrs = [
        make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state=h)
        for h in hmm.hidden_states
    ]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    with pytest.raises(InvalidInputError, match="No feasible augmented path"):
        marginal_map_torch_mvr_chmm(
            model, observed, query_times=[0, 4], return_augmented=False
        )


def test_gap_operator_allocation_failure_names_the_cause(hmm, observed):
    from conin.hidden_markov_model.inference import marginal_map_mvr as mod

    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    original = mod._sum_step
    mod._sum_step = lambda *args, **kwargs: (_ for _ in ()).throw(MemoryError())
    try:
        with pytest.raises(InvalidInputError, match="MVRs \\[0\\] span this gap"):
            marginal_map_torch_mvr_chmm(
                model, observed, query_times=[0, 3], return_augmented=False
            )
    finally:
        mod._sum_step = original


# ===========================
# Numerical stability
# ===========================


@pytest.mark.parametrize("gap", [1500])
def test_long_constrained_gap_does_not_underflow(gap):
    """
    A constraint held across a long summed-out gap has exponentially small
    satisfying mass. In probability space the gap operator flushes to subnormals
    and the score silently saturates at ``log(smallest subnormal)``; in log space
    float32 must still track float64.
    """
    hmm = make_random_hmm(
        hidden_states=["A", "B", "C"], observed_states=["o0", "o1"], seed=7
    )
    T = gap + 1
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    scores = {}
    for dt in (torch.float32, torch.float64):
        _, scores[dt] = marginal_map_torch_mvr_chmm(
            model, {}, time_horizon=T, query_times=[0, T - 1],
            dtype=dt, return_augmented=False, return_score=True,
        )

    # Well past every float32 floor: log(min subnormal) is about -103.
    assert scores[torch.float64] < -110

    # Agreement to within ordinary float32 drift, which grows with the step count.
    assert scores[torch.float32] == pytest.approx(
        scores[torch.float64], rel=1e-4, abs=1e-3 * gap
    )


def test_gap_operator_propagates_non_allocation_errors(hmm, observed):
    """A real bug must not be relabelled as a memory problem."""
    from conin.hidden_markov_model.inference import marginal_map_mvr as mod

    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    boom = RuntimeError("some unrelated shape bug")
    original = mod._sum_step
    mod._sum_step = lambda *args, **kwargs: (_ for _ in ()).throw(boom)
    try:
        with pytest.raises(RuntimeError, match="some unrelated shape bug"):
            marginal_map_torch_mvr_chmm(
                model, observed, query_times=[0, 3], return_augmented=False
            )
    finally:
        mod._sum_step = original


