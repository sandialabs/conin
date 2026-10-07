import copy
import itertools
import warnings

import numpy as np
import pytest

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.chmm_mvr import MVR_CHMM
from conin.hidden_markov_model.mvr_constraints import mvr_current_state
from conin.hidden_markov_model.mvr_operators import mvr_count, mvr_timerange

torch = pytest.importorskip("torch")

from conin.hidden_markov_model.learning.generalized_em_mvr import (  # noqa: E402
    _chain_gradient,
    _constraint_statistics,
    generalized_em_mvr_chmm,
)
from conin.hidden_markov_model.mvr_common import (  # noqa: E402
    _build_sumprod_ctx,
    _hmm_to_torch,
)
from .test_viterbi_mvr import (  # noqa: E402
    as_obs_map,
    make_end_state_inhom_mvr,
    make_random_hmm,
    mvr_accepts,
    score_path,
)


def enumerate_objectives(hmm, constraints, observations, horizons, posterior=None):
    likelihood, surrogate, weights = 0.0, 0.0, []
    counts = [
        np.zeros_like(p) for p in (hmm.start_vec, hmm.transition_mat, hmm.emission_mat)
    ]

    def score(path, observed):
        try:
            return score_path(hmm, path, as_obs_map(observed))
        except ValueError:
            return -np.inf

    for i, (observed, horizon) in enumerate(zip(observations, horizons)):
        paths = [
            list(p)
            for p in itertools.product(hmm.hidden_states, repeat=horizon)
            if all(mvr_accepts(c, p, horizon) for c in constraints)
        ]
        scores = np.array([score(p, observed) for p in paths])
        prior = np.array([score(p, {}) for p in paths])
        joint, normalizer = np.logaddexp.reduce(scores), np.logaddexp.reduce(prior)
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
    return likelihood, surrogate, weights, counts


def make_case(seed):
    hmm = make_random_hmm(
        hidden_states=["C", "A", "B"], observed_states=["y", "x"], seed=seed
    )
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
        constraints.append(
            mvr_timerange(mvr_count(mvr_current_state(hmm, {"B"}), ">=1"), [1, 2])
        )
    if seed % 3 == 2:
        constraints.append(
            make_end_state_inhom_mvr(
                hidden_states=hmm.hidden_states,
                target_state="C",
                time_horizon=1,
                time_range=[2, 3],
            )
        )
    observations = [["x", "y", "x", "y"], {0: "y", 2: "x"}, {}]
    horizons = [4, 5, 4]
    if seed == 0:
        observations, horizons = [["x"], {0: "y"}, {}], [1, 2, 1]
    update = ("start", "transition", "emission") if seed % 2 == 0 else ("transition",)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=constraints)
    return hmm, model, observations, horizons, update


@pytest.mark.parametrize("seed", range(6))
def test_gem_matches_enumeration(seed):
    hmm, model, observations, horizons, update = make_case(seed)
    constraints = model.constraints
    caller, original = hmm, copy.deepcopy(hmm)
    previous = enumerate_objectives(hmm, constraints, observations, horizons)[0]
    initial = previous
    for _ in range(3):
        _, old_q, posterior, counts = enumerate_objectives(
            hmm, constraints, observations, horizons
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Initial model has zero-probability"
            )
            fitted, history = generalized_em_mvr_chmm(
                MVR_CHMM(hidden_markov_model=hmm, constraints=constraints),
                observations,
                time_horizons=horizons,
                max_iter=1,
                tol=0,
                pseudocount=0,
                update=update,
                inner_max_iter=3,
            )
        score, new_q, _, _ = enumerate_objectives(
            fitted, constraints, observations, horizons, posterior
        )
        assert history == pytest.approx([previous], abs=1e-9)
        assert score >= previous - 1e-9
        assert new_q >= old_q - 1e-9
        if "emission" in update:
            expected = np.asarray(hmm.emission_mat).copy()
            totals = counts[2].sum(axis=1, keepdims=True)
            occupied = totals[:, 0] > 0
            expected[occupied] = counts[2][occupied] / totals[occupied]
            assert np.asarray(fitted.emission_mat) == pytest.approx(expected, abs=1e-9)
        for name, attr in zip(
            ("start", "transition", "emission"),
            ("start_vec", "transition_mat", "emission_mat"),
        ):
            before, after = np.asarray(getattr(hmm, attr)), np.asarray(
                getattr(fitted, attr)
            )
            assert after[before == 0] == pytest.approx(0)
            if name not in update:
                assert np.array_equal(before, after)
        hmm, previous = fitted, score
    assert previous > initial + 1e-6
    for attr in ("start_vec", "transition_mat", "emission_mat"):
        assert getattr(caller, attr) == getattr(original, attr)


@pytest.mark.parametrize("seed", range(6))
def test_chain_gradient_matches_finite_differences(seed):
    # The ascent checks above would also pass for Baum-Welch; this pins the Z term.
    hmm, model, observations, horizons, update = make_case(seed)
    constraints = model.constraints
    _, _, posterior, counts = enumerate_objectives(
        hmm, constraints, observations, horizons
    )
    logs = list(_hmm_to_torch(hmm, log=True, dtype=torch.float64))
    contexts = {
        t: _build_sumprod_ctx(model, {}, time_horizon=t, dtype=torch.float64)
        for t in set(horizons)
    }
    multiplicities = {t: horizons.count(t) for t in set(horizons)}
    prior, _ = _constraint_statistics(logs, contexts, multiplicities, counts=True)
    gradients = _chain_gradient(
        logs, [torch.tensor(c) / 3 for c in counts], [c / 3 for c in prior], update
    )
    for block, (name, attr) in enumerate(
        zip(("start", "transition"), ("start_vec", "transition_mat"))
    ):
        if name not in update:
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
                values.append(
                    enumerate_objectives(
                        perturbed, constraints, observations, horizons, posterior
                    )[1]
                    / 3
                )
            assert float(gradients[block][index]) == pytest.approx(
                (values[1] - values[0]) / 2e-5, abs=1e-8
            )


def test_pseudocount_keeps_unobserved_emissions_positive():
    hmm = make_random_hmm(
        hidden_states=["A", "B"], observed_states=["x", "y", "z"], seed=3
    )
    hmm.emission_mat[0] = [0.5, 0.5, 0.0]
    hmm.initialize(avoid_reinitialization=False)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    with pytest.warns(UserWarning, match="emission"):
        fitted, _ = generalized_em_mvr_chmm(model, [["x", "y", "x"]], max_iter=1, tol=0)
    unobserved = np.asarray(fitted.emission_mat)[:, hmm.observed_to_internal["z"]]
    assert unobserved[0] == 0 and unobserved[1] > 0


def test_gem_history():
    # Deliberate executable spec: history follows Baum–Welch.
    hmm = make_random_hmm(hidden_states=["A", "B"], observed_states=["x", "y"], seed=8)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    observations = [["x", "x", "y"]]
    fitted, history = generalized_em_mvr_chmm(model, observations, max_iter=2, tol=0)
    assert len(history) == 2
    assert enumerate_objectives(fitted, [], observations, [3])[0] >= history[-1]
    fitted, history = generalized_em_mvr_chmm(model, [{}], time_horizons=3)
    assert history == pytest.approx([0, 0], abs=1e-12)
    fitted, history = generalized_em_mvr_chmm(model, observations, max_iter=0)
    assert history == []
    assert fitted.emission_mat == hmm.emission_mat


def test_gem_backtracking_failure_retains_parameters():
    hmm = make_random_hmm(hidden_states=["A", "B"], observed_states=["x", "y"], seed=8)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    with pytest.warns(RuntimeWarning, match="backtracking"):
        fitted, history = generalized_em_mvr_chmm(
            model,
            [["x", "x", "y"]],
            max_iter=1,
            update=("start", "transition"),
            step_size=1e10,
            max_backtracks=1,
        )
    assert history == pytest.approx([history[0], history[0]])
    assert fitted.start_vec == pytest.approx(hmm.start_vec)
    assert np.asarray(fitted.transition_mat) == pytest.approx(
        np.asarray(hmm.transition_mat)
    )


@pytest.mark.parametrize(
    "observations, kwargs",
    [
        ([["x"]], {"update": ("start", "bogus")}),
        ([["x"]], {"pseudocount": -1.0}),
        ([], {}),
    ],
    ids=["update", "budget", "empty"],
)
def test_gem_rejects_invalid_arguments(observations, kwargs):
    hmm = make_random_hmm(hidden_states=["A", "B"], observed_states=["x", "y"], seed=8)
    model = MVR_CHMM(hidden_markov_model=hmm, constraints=[])
    with pytest.raises(InvalidInputError):
        generalized_em_mvr_chmm(model, observations, **kwargs)
