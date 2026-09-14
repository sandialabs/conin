import copy
import itertools
import warnings

import numpy as np
import pytest

from conin.hidden_markov_model.chmm_mvr import MVR_CHMM
from conin.hidden_markov_model.mvr_constraints import mvr_current_state
from conin.hidden_markov_model.mvr_operators import mvr_count, mvr_timerange

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.learning.generalized_em_mvr import (  # noqa: E402
    _chain_gradient,
    _constraint_statistics,
    generalized_em_mvr_chmm,
)
from conin.hidden_markov_model.mvr_common import _build_sumprod_ctx, _hmm_to_torch  # noqa: E402
from .test_viterbi_mvr import (  # noqa: E402
    as_obs_map,
    make_end_state_inhom_mvr,
    make_random_hmm,
    mvr_accepts,
    score_path,
)


def enumerate_objectives(hmm, constraints, observations, horizons, posterior=None):
    likelihood, surrogate, weights = 0.0, 0.0, []
    log_z = 0.0
    counts = [np.zeros_like(p) for p in (hmm.start_vec, hmm.transition_mat, hmm.emission_mat)]

    def score(path, observed):
        try:
            return score_path(hmm, path, as_obs_map(observed))
        except ValueError:
            return -np.inf

    for i, (observed, horizon) in enumerate(zip(observations, horizons)):
        paths = [list(p) for p in itertools.product(hmm.hidden_states, repeat=horizon)
                 if all(mvr_accepts(c, p, horizon) for c in constraints)]
        scores = np.array([score(p, observed) for p in paths])
        prior = np.array([score(p, {}) for p in paths])
        joint, normalizer = np.logaddexp.reduce(scores), np.logaddexp.reduce(prior)
        log_z += normalizer
        weights.append(np.exp(scores - joint))
        q = weights[-1] if posterior is None else posterior[i]
        positive = q > 0
        surrogate += q[positive] @ scores[positive] - normalizer
        likelihood += joint - normalizer
        for path, weight in zip(paths, weights[-1]):
            states = [hmm.hidden_to_internal[h] for h in path]
            counts[0][states[0]] += weight
            for left, right in zip(states, states[1:]):
                counts[1][left, right] += weight
            for t, label in as_obs_map(observed).items():
                counts[2][states[t], hmm.observed_to_internal[label]] += weight
    return likelihood, surrogate, weights, counts, log_z


@pytest.mark.parametrize("seed", range(6))
def test_gem_matches_enumeration(seed):
    hmm = make_random_hmm(hidden_states=["C", "A", "B"], observed_states=["y", "x"], seed=seed)
    if seed == 4:
        hmm.start_vec[1] = 0.0
        start = np.asarray(hmm.start_vec)
        hmm.start_vec = (start / start.sum()).tolist()
        hmm.transition_mat[0][1] = 0.0
        row = np.asarray(hmm.transition_mat[0])
        hmm.transition_mat[0] = (row / row.sum()).tolist()
        hmm.initialize(avoid_reinitialization=False)
    constraints = []
    if seed % 3:
        constraints.append(mvr_timerange(
            mvr_count(mvr_current_state(hmm, {"B"}), ">=1"), [1, 2]
        ))
    if seed % 3 == 2:
        constraints.append(make_end_state_inhom_mvr(
            hidden_states=hmm.hidden_states, target_state="C", time_horizon=1,
            time_range=[2, 3],
        ))
    observations = [["x", "y", "x", "y"], {0: "y", 2: "x"}, {}]
    horizons = [4, 5, 4]
    if seed == 0:
        observations, horizons = [["x"], {0: "y"}, {}], [1, 2, 1]
    update = ("start", "transition", "emission") if seed % 2 == 0 else ("transition",)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=constraints)
    constraints = model.constraints
    original = copy.deepcopy(hmm)
    initial, _, posterior, counts, _ = enumerate_objectives(hmm, constraints, observations, horizons)
    logs = list(_hmm_to_torch(hmm, log=True, dtype=torch.float64))
    contexts = {t: _build_sumprod_ctx(model, {}, time_horizon=t, dtype=torch.float64) for t in set(horizons)}
    multiplicities = {t: horizons.count(t) for t in set(horizons)}
    prior, normalizer = _constraint_statistics(logs, contexts, multiplicities, counts=True)
    reference = enumerate_objectives(hmm, constraints, [{}, {}, {}], horizons)
    assert float(normalizer) == pytest.approx(reference[4], abs=1e-9)
    for actual, expected in zip(prior, reference[3][:2]):
        assert actual.numpy() == pytest.approx(expected, abs=1e-9)
    gradients = _chain_gradient(logs, [torch.tensor(c) / 3 for c in counts[:3]],
                                [c / 3 for c in prior], update)
    for block, attr in enumerate(("start_vec", "transition_mat")):
        if ("start", "transition")[block] not in update:
            continue
        for index in np.ndindex(logs[block].shape):
            if not torch.isfinite(logs[block][index]):
                continue
            values = []
            for delta in (-1e-5, 1e-5):
                perturbed = copy.deepcopy(hmm)
                logits = logs[block].clone()
                logits[index] += delta
                setattr(perturbed, attr, logits.softmax(-1).tolist())
                perturbed.initialize(avoid_reinitialization=False)
                values.append(enumerate_objectives(
                    perturbed, constraints, observations, horizons, posterior
                )[1] / 3)
            assert float(gradients[block][index]) == pytest.approx((values[1] - values[0]) / 2e-5, abs=1e-8)
    previous = initial
    for _ in range(3):
        _, old_q, posterior, counts, _ = enumerate_objectives(hmm, constraints, observations, horizons)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initial model has zero-probability")
            fitted, history = generalized_em_mvr_chmm(
                MVR_CHMM(hidden_markov_model=hmm, constraints=constraints), observations,
                time_horizons=horizons, max_iter=1, tol=0, update=update, inner_max_iter=3,
            )
        score, new_q, _, _, _ = enumerate_objectives(fitted, constraints, observations, horizons, posterior)
        assert history == pytest.approx([previous], abs=1e-9)
        assert score >= previous - 1e-9
        assert new_q >= old_q - 1e-9
        if "emission" in update:
            expected = np.asarray(hmm.emission_mat).copy()
            occupied = counts[2].sum(axis=1) > 0
            expected[occupied] = counts[2][occupied] / counts[2][occupied].sum(axis=1, keepdims=True)
            assert np.asarray(fitted.emission_mat) == pytest.approx(expected, abs=1e-9)
        for name, attr in zip(("start", "transition", "emission"), ("start_vec", "transition_mat", "emission_mat")):
            before, after = np.asarray(getattr(hmm, attr)), np.asarray(getattr(fitted, attr))
            assert after[before == 0] == pytest.approx(0)
            assert after.sum(axis=-1) == pytest.approx(1)
            if name not in update:
                assert np.array_equal(before, after)
        hmm, previous = fitted, score
    assert previous > initial + 1e-6
    for attr in ("start_vec", "transition_mat", "emission_mat", "hidden_to_external", "observed_to_external"):
        assert getattr(model.hidden_markov_model, attr) == getattr(original, attr)
    assert fitted.hidden_to_external == original.hidden_to_external
    assert fitted.observed_to_external == original.observed_to_external


def test_gem_history_and_backtracking():
    # Deliberate executable spec: history follows Baum–Welch except on failed search.
    hmm = make_random_hmm(hidden_states=["A", "B"], observed_states=["x", "y"], seed=8)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    observations = [["x", "x", "y"]]
    fitted, history = generalized_em_mvr_chmm(model, observations, max_iter=2, tol=0)
    assert len(history) == 2
    assert enumerate_objectives(fitted, [], observations, [3])[0] >= history[-1]
    fitted, history = generalized_em_mvr_chmm(model, [{}], time_horizons=3)
    assert history == pytest.approx([0, 0], abs=1e-12)
    with pytest.warns(RuntimeWarning, match="GEM backtracking failed"):
        fitted, history = generalized_em_mvr_chmm(
            model, observations, max_iter=1, update=("start", "transition"),
            step_size=1e10, max_backtracks=1,
        )
    assert history == pytest.approx([history[0], history[0]])
    assert fitted.start_vec == pytest.approx(hmm.start_vec)
    assert np.asarray(fitted.transition_mat) == pytest.approx(np.asarray(hmm.transition_mat))
