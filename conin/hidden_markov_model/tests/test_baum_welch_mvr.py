import itertools
import math
import warnings

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM
from conin.hidden_markov_model.hmm import HiddenMarkovModel

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.learning.baum_welch_mvr import (  # noqa: E402
    baum_welch_mvr_chmm,
    forward_backward_mvr_chmm,
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
# Reference implementation
# ===========================


def brute_force_posteriors(hmm, mvrs, observed, T):
    """
    Exact posteriors by enumeration: weight every feasible path, normalize, and
    accumulate the marginals. This is the definition rather than a second
    implementation of the recursion.
    """
    obs_map = as_obs_map(observed)
    hidden_states = list(hmm.hidden_states)
    index = {h: i for i, h in enumerate(hidden_states)}
    K = len(hidden_states)

    paths, scores = [], []

    for path in itertools.product(hidden_states, repeat=T):
        if all(mvr_accepts(mvr, path, T) for mvr in mvrs):
            paths.append(path)
            scores.append(score_path(hmm, list(path), obs_map))

    if not paths:
        return None, None, -math.inf

    scores = np.asarray(scores)
    peak = scores.max()
    loglik = float(np.log(np.exp(scores - peak).sum()) + peak)

    weights = np.exp(scores - loglik)

    gamma = np.zeros((T, K))
    xi = np.zeros((max(T - 1, 0), K, K))

    for weight, path in zip(weights, paths):
        for t in range(T):
            gamma[t, index[path[t]]] += weight
        for t in range(T - 1):
            xi[t, index[path[t]], index[path[t + 1]]] += weight

    return gamma, xi, loglik


def brute_force_counts(hmm, mvrs, observations, horizons):
    """Pooled expected sufficient statistics, straight from the enumeration."""
    K = hmm.num_hidden_states
    num_observed = hmm.num_observed_states

    init_counts = np.zeros(K)
    trans_counts = np.zeros((K, K))
    emit_counts = np.zeros((K, num_observed))
    total_loglik = 0.0

    for observed, T in zip(observations, horizons):
        gamma, xi, loglik = brute_force_posteriors(hmm, mvrs, observed, T)

        init_counts += gamma[0]
        trans_counts += xi.sum(axis=0)
        total_loglik += loglik

        for t, o in as_obs_map(observed).items():
            emit_counts[:, hmm.observed_to_internal[o]] += gamma[t]

    return init_counts, trans_counts, emit_counts, total_loglik


def assert_matches_brute_force(hmm, mvrs, observed, T=None):
    if T is None:
        T = len(observed)

    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    gamma, xi, loglik = forward_backward_mvr_chmm(model, observed, time_horizon=T)
    expected_gamma, expected_xi, expected_loglik = brute_force_posteriors(
        hmm, mvrs, observed, T
    )

    assert loglik == pytest.approx(expected_loglik, abs=1e-9)
    assert gamma.numpy() == pytest.approx(expected_gamma, abs=1e-9)
    assert xi.numpy() == pytest.approx(expected_xi, abs=1e-9)

    return gamma, xi, loglik


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
# Posteriors against brute force
# ===========================


def test_forward_backward_inhom_mvr_windowed(hmm, observed):
    # The random sweep below is homogeneous only, so this is the one place the
    # local-time offsets (t - a for a slice, t - a - 1 for a transition) run
    # with a nonzero a.
    mvr = make_end_state_inhom_mvr(
        hidden_states=hmm.hidden_states,
        target_state="C",
        time_horizon=2,
        time_range=[1, 3],
    )
    assert_matches_brute_force(hmm, [mvr], observed)


def test_forward_backward_overlapping_windows_mixed_types(hmm, observed):
    mvrs = [
        make_forbid_mvr(
            hidden_states=hmm.hidden_states, forbidden_state="A", time_range=[0, 2]
        ),
        make_parity_mvr(hidden_states=hmm.hidden_states, target_state="B"),
        make_end_state_inhom_mvr(
            hidden_states=hmm.hidden_states,
            target_state="C",
            time_horizon=2,
            time_range=[2, 4],
        ),
    ]
    assert_matches_brute_force(hmm, mvrs, observed)


def test_forward_backward_single_time_step(hmm):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    gamma, xi, _ = assert_matches_brute_force(hmm, [mvr], ["o1"])

    assert gamma.shape == (1, 3)
    assert xi.shape == (0, 3, 3)


@pytest.mark.parametrize("seed", range(12))
def test_forward_backward_random_instances(seed):
    # The core correctness test. Across these seeds the draw covers an empty
    # constraint set, a defaulted time_range, and windows that start at 0, end
    # at T-1, sit in the interior, and collapse to a single time -- which is why
    # none of those has a standalone test.
    rng = np.random.default_rng(seed)

    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]

    model_hmm = make_random_hmm(
        hidden_states=hidden_states, observed_states=observed_states, seed=seed
    )
    T = int(rng.integers(2, 7))
    observed = [observed_states[i] for i in rng.integers(0, 2, size=T)]

    mvrs = []

    if seed % 3:
        start = int(rng.integers(0, T))
        mvrs.append(
            make_forbid_mvr(
                hidden_states=hidden_states,
                forbidden_state="A",
                time_range=[start, int(rng.integers(start, T))],
            )
        )
    if seed % 2:
        mvrs.append(make_parity_mvr(hidden_states=hidden_states, target_state="B"))

    assert_matches_brute_force(model_hmm, mvrs, observed)


# ===========================
# Horizon and sparse observations
# ===========================


def test_forward_backward_sparse_observations(hmm):
    # Sparse map plus a horizon past the last observation. The other horizon
    # forms belong to _resolve_horizon, which test_viterbi_mvr already covers.
    observed = {0: "o1", 3: "o0"}
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states, forbidden_state="A", time_range=[1, 4]
    )
    assert_matches_brute_force(hmm, [mvr], observed, T=5)


# ===========================
# Internal consistency
# ===========================


def test_augmented_posterior_marginalizes_to_gamma(hmm, observed):
    mvrs = [
        make_forbid_mvr(
            hidden_states=hmm.hidden_states, forbidden_state="A", time_range=[0, 2]
        ),
        make_parity_mvr(hidden_states=hmm.hidden_states, target_state="B"),
    ]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    gamma, _, _, augmented = forward_backward_mvr_chmm(
        model, observed, return_augmented=True
    )

    assert len(augmented) == len(observed)

    for t, slice_t in enumerate(augmented):
        # Only the MVRs active at t carry an axis, so the shape is dynamic.
        expected_axes = 1 + len(mvrs) if t <= 2 else 2
        assert slice_t.dim() == expected_axes
        assert slice_t.reshape(3, -1).sum(dim=1).numpy() == pytest.approx(
            gamma[t].numpy(), abs=1e-9
        )


def test_forward_backward_raises_on_infeasible_constraints(hmm, observed):
    mvrs = [
        make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state=h)
        for h in hmm.hidden_states
    ]
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=mvrs)

    with pytest.raises(InvalidInputError, match="No feasible augmented path"):
        forward_backward_mvr_chmm(model, observed)


# ===========================
# Baum-Welch
# ===========================


def make_batch(observed_states, *, count, length, seed):
    rng = np.random.default_rng(seed)

    return [
        [observed_states[i] for i in rng.integers(0, len(observed_states), size=length)]
        for _ in range(count)
    ]


def test_one_em_step_matches_brute_force_counts(hmm):
    # The M step is the normalization of the expected counts, so reproducing the
    # updated parameters from enumerated counts pins the whole iteration.
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states, forbidden_state="A", time_range=[0, 2]
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    observations = make_batch(["o0", "o1"], count=3, length=5, seed=1)

    # One iteration is one E step followed by one M step.
    fitted, history = baum_welch_mvr_chmm(
        model, observations, max_iter=1, tol=0.0, pseudocount=0.0
    )

    init_counts, trans_counts, emit_counts, loglik = brute_force_counts(
        hmm, [mvr], observations, [5] * len(observations)
    )

    assert history[0] == pytest.approx(loglik, abs=1e-9)
    assert np.asarray(fitted.start_vec) == pytest.approx(
        init_counts / init_counts.sum(), abs=1e-9
    )
    assert np.asarray(fitted.transition_mat) == pytest.approx(
        trans_counts / trans_counts.sum(axis=1, keepdims=True), abs=1e-9
    )
    assert np.asarray(fitted.emission_mat) == pytest.approx(
        emit_counts / emit_counts.sum(axis=1, keepdims=True), abs=1e-9
    )


def test_em_keeps_rows_that_collect_no_mass(hmm, observed):
    # Forbidding A outright makes it unreachable, so nothing ever leaves it and
    # nothing is ever emitted from it. Those rows have no maximizer to find and
    # must be left exactly as the caller supplied them.
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    fitted, _ = baum_welch_mvr_chmm(model, [observed], max_iter=5, tol=0.0)

    a = hmm.hidden_to_internal["A"]

    assert np.asarray(fitted.transition_mat)[a] == pytest.approx(
        np.asarray(hmm.transition_mat)[a]
    )
    assert np.asarray(fitted.emission_mat)[a] == pytest.approx(
        np.asarray(hmm.emission_mat)[a]
    )
    assert fitted.start_vec[a] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("seed", range(3))
def test_em_log_likelihood_is_monotone(seed):
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]

    start_hmm = make_random_hmm(
        hidden_states=hidden_states, observed_states=observed_states, seed=seed
    )
    mvrs = [
        make_forbid_mvr(
            hidden_states=hidden_states, forbidden_state="A", time_range=[0, 3]
        ),
        make_parity_mvr(hidden_states=hidden_states, target_state="B"),
    ]
    model = MVR_CHMM(hidden_markov_model=start_hmm, constraints=mvrs)

    observations = make_batch(observed_states, count=4, length=6, seed=seed)

    _, history = baum_welch_mvr_chmm(model, observations, max_iter=25, tol=0.0)

    assert len(history) == 25
    # Exact EM on this objective, so any decrease is a bug rather than noise.
    assert np.diff(history).min() >= -1e-9


def test_em_handles_mixed_lengths_and_observation_forms():
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]

    start_hmm = make_random_hmm(
        hidden_states=hidden_states, observed_states=observed_states, seed=5
    )
    mvr = make_forbid_mvr(
        hidden_states=hidden_states, forbidden_state="A", time_range=[0, 2]
    )
    model = MVR_CHMM(hidden_markov_model=start_hmm, constraints=[mvr])

    observations = [["o0", "o1", "o0", "o1"], {0: "o1", 3: "o0"}, ["o1"] * 6]

    _, history = baum_welch_mvr_chmm(
        model, observations, time_horizons=[None, 5, None], max_iter=12, tol=0.0
    )

    assert np.diff(history).min() >= -1e-9


def test_em_leaves_the_caller_model_untouched(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    before = np.asarray(hmm.transition_mat).copy()

    fitted, _ = baum_welch_mvr_chmm(model, [observed], max_iter=5, tol=0.0)

    assert np.asarray(hmm.transition_mat) == pytest.approx(before)
    assert fitted is not hmm
    assert not np.allclose(np.asarray(fitted.transition_mat), before)


def test_em_write_back_preserves_indexing_and_rebuilds_repn(hmm, observed):
    # load_model re-derives the internal order from sorted keys, so a write-back
    # through it could permute indices out from under the aligned constraints.
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    fitted, _ = baum_welch_mvr_chmm(model, [observed], max_iter=3, tol=0.0)

    assert fitted.hidden_to_internal == hmm.hidden_to_internal
    assert fitted.observed_to_internal == hmm.observed_to_internal
    assert fitted.hidden_to_external == hmm.hidden_to_external
    assert fitted.observed_to_external == hmm.observed_to_external

    # A stale repn would leave inference running on the pre-fit parameters.
    # Row sums are not checked: HMM_MatVecRepn validates those on initialize.
    assert fitted.repn.transition_mat == fitted.transition_mat


def test_em_stops_at_the_convergence_tolerance(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    fitted, history = baum_welch_mvr_chmm(model, [observed], max_iter=200, tol=1e-8)

    assert len(history) < 200
    assert abs(history[-1] - history[-2]) < 1e-8

    # Converged, so no update followed the last entry: it scores exactly the
    # model that came back.
    fitted_model = MVR_CHMM(hidden_markov_model=fitted, constraints=[mvr])
    assert forward_backward_mvr_chmm(fitted_model, observed)[2] == pytest.approx(
        history[-1], abs=1e-9
    )


def test_em_history_starts_at_the_input_model(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    # A tolerance this tight cannot be met in three iterations, so the run also
    # exercises the not-converged warning.
    with pytest.warns(UserWarning, match="without reaching tol"):
        _, history = baum_welch_mvr_chmm(model, [observed], max_iter=3, tol=1e-14)

    assert history[0] == pytest.approx(
        forward_backward_mvr_chmm(model, observed)[2], abs=1e-9
    )
    assert len(history) == 3


@pytest.mark.parametrize(
    "block, attribute",
    [
        ("start", "start_vec"),
        ("transition", "transition_mat"),
        ("emission", "emission_mat"),
    ],
)
def test_em_updates_only_the_requested_blocks(hmm, observed, block, attribute):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    fitted, _ = baum_welch_mvr_chmm(
        model, [observed], max_iter=3, tol=0.0, update=(block,)
    )

    for name in ("start_vec", "transition_mat", "emission_mat"):
        unchanged = np.allclose(
            np.asarray(getattr(fitted, name)), np.asarray(getattr(hmm, name))
        )
        assert bool(unchanged) == (name != attribute)


# ===========================
# Structural zeros
# ===========================


def make_sparse_hmm(hidden_states, observed_states):
    """A model with a forbidden transition and a state that emits one symbol."""
    hmm = HiddenMarkovModel()

    transition = {
        (h1, h2): (0.0 if (h1, h2) == ("A", "C") else 0.5) for h1 in hidden_states
        for h2 in hidden_states
    }
    for h1 in hidden_states:
        total = sum(transition[h1, h2] for h2 in hidden_states)
        for h2 in hidden_states:
            transition[h1, h2] /= total

    emission = {
        (h, o): (1.0 if o == observed_states[0] else 0.0) if h == "C" else 0.5
        for h in hidden_states
        for o in observed_states
    }

    hmm.load_model(
        start_probs={h: 1.0 / len(hidden_states) for h in hidden_states},
        transition_probs=transition,
        emission_probs=emission,
        initialize=True,
    )

    return hmm


def test_em_preserves_structural_zeros_and_warns():
    hidden_states = ["A", "B", "C"]
    observed_states = ["o0", "o1"]

    sparse = make_sparse_hmm(hidden_states, observed_states)
    mvr = make_forbid_mvr(hidden_states=hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=sparse, constraints=[mvr])

    observations = make_batch(observed_states, count=3, length=5, seed=2)

    with pytest.warns(UserWarning, match="zero-probability entries"):
        fitted, _ = baum_welch_mvr_chmm(model, observations, max_iter=15, tol=0.0)

    assert np.asarray(fitted.transition_mat)[0, 2] == 0.0
    assert np.asarray(fitted.emission_mat)[2, 1] == 0.0


def test_dense_model_does_not_warn(hmm, observed):
    mvr = make_forbid_mvr(hidden_states=hmm.hidden_states, forbidden_state="A")
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        baum_welch_mvr_chmm(model, [observed], max_iter=2, tol=0.0)


# ===========================
# Validation
# ===========================


def test_em_names_the_offending_sequence(hmm):
    mvr = make_forbid_mvr(
        hidden_states=hmm.hidden_states, forbidden_state="A", time_range=[0, 4]
    )
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[mvr])

    with pytest.raises(InvalidInputError, match="Sequence 1"):
        baum_welch_mvr_chmm(model, [["o0"] * 5, ["o1"] * 3], max_iter=1)


@pytest.mark.parametrize(
    "kwargs, observations, message",
    [
        ({}, [], "at least one sequence"),
        ({"update": ("start", "bogus")}, [["o0"]], "Unknown update targets"),
        ({"time_horizons": [5, 5]}, [["o0"]], "entries but there are"),
    ],
)
def test_em_rejects_bad_arguments(hmm, kwargs, observations, message):
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])

    with pytest.raises(InvalidInputError, match=message):
        baum_welch_mvr_chmm(model, observations, max_iter=1, **kwargs)
