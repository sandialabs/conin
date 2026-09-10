import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from munch import Munch
from scipy.special import logsumexp

torch = pytest.importorskip('torch')
import dna_algorithms as dna


def make_hmm(seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(2))
    a = rng.dirichlet(np.ones(2), size=2)
    b = rng.dirichlet(np.ones(2), size=2)
    if seed == 2:
        a[0] = [0., 1.]
        b[1] = [1., 0.]
    if seed == 3:
        p = np.array([1., 1e-250])
        a[:, :] = [1., 1e-250]
    return Munch(states=[0, 1], emits=[0, 1],
                 initprob=dict(enumerate(p)),
                 tprob={(i, j): a[i, j] for i in range(2) for j in range(2)},
                 eprob={(i, j): b[i, j] for i in range(2) for j in range(2)})


def constraints(kind):
    first = Munch(m_states=[False, True],
                  init_fun=lambda k, r: r == (k == 0),
                  update_fun=lambda k, r, prev: r == (prev or k == 0),
                  eval_fun=lambda k, r: r)
    odd = Munch(m_states=[0, 1], init_fun=lambda k, r: r == k,
                update_fun=lambda k, r, prev: r == (prev + k) % 2,
                eval_fun=lambda k, r: r == 1)
    rare = Munch(m_states=[0], init_fun=lambda k, r: k == 1,
                 update_fun=lambda k, r, prev: k == 1,
                 eval_fun=lambda k, r: True)
    if kind == 'none':
        return []
    if kind == 'rare':
        return [rare]
    return [first, odd]


def enumerate_model(hmm, horizon, kind, deadline, obs=None):
    paths, scores = [], []
    for x in itertools.product(hmm.states, repeat=horizon):
        if kind == 'hit_odd' and not (0 in x[:deadline] and sum(x) % 2 == 1):
            continue
        if kind == 'rare' and any(k != 1 for k in x):
            continue
        factors = [hmm.initprob[x[0]]] + [hmm.tprob[i, j] for i, j in zip(x, x[1:])]
        if obs is not None:
            factors += [hmm.eprob[k, e] for k, e in zip(x, obs)]
        if any(v == 0 for v in factors):
            continue
        paths.append(x)
        scores.append(sum(np.log(v) for v in factors))
    if not scores:
        return [], np.array([]), -np.inf, None
    z = logsumexp(scores)
    weights = np.exp(np.array(scores) - z)
    counts = [np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))]
    for x, w in zip(paths, weights):
        counts[0][x[0]] += w
        for i, j in zip(x, x[1:]):
            counts[1][i, j] += w
        if obs is not None:
            for k, e in zip(x, obs):
                counts[2][k, e] += w
    return paths, weights, z, counts


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_counts_and_sampling_match_enumeration(seed, device):
    hmm = make_hmm(seed)
    for horizon in range(1, 6):
        deadline = min(2, horizon)
        for kind in ['none', 'hit_odd', 'rare']:
            csts = constraints(kind)
            for obs in [None, *itertools.product(hmm.emits, repeat=horizon)]:
                _, _, score, expected = enumerate_model(hmm, horizon, kind, deadline, obs)
                if not np.isfinite(score):
                    with pytest.raises(ValueError, match='zero probability'):
                        dna.constrained_e_step_counts(hmm, csts, obs,
                            pro_before=deadline, time_horizon=horizon, device=device)
                    continue
                *actual, actual_score = dna.constrained_e_step_counts(
                    hmm, csts, obs, pro_before=deadline, time_horizon=horizon, device=device)
                assert actual_score == pytest.approx(score, abs=2e-10)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a.cpu().numpy(), b, atol=2e-11)
            paths, weights, _, _ = enumerate_model(hmm, horizon, kind, deadline)
            for path, weight in zip(paths, weights):
                dims = tuple(len(c.m_states) for c in csts)
                memories, previous = [], None
                for t, k in enumerate(path):
                    current = []
                    for i, c in enumerate(csts):
                        accepted = [j for j, r in enumerate(c.m_states)
                                    if (c.init_fun(k, r) if t == 0 else
                                        c.update_fun(k, r, c.m_states[previous[i]]))]
                        current.append(accepted[0])
                    memories.append(np.ravel_multi_index(tuple(current), dims) if dims else 0)
                    previous = current
                size = int(np.prod(dims))
                targets = [k * size + m for k, m in zip(path, memories)][::-1]

                class ForcedDraws:
                    def __init__(self):
                        self.calls, self.probability = 0, 1.

                    def choice(self, n, p):
                        result = targets[self.calls] if self.calls < horizon else 0
                        if self.calls < horizon:
                            self.probability *= p[result]
                        self.calls += 1
                        return result

                rng = ForcedDraws()
                sampled, = dna.sample_constrained_fixed_length(
                    hmm, csts, horizon, pro_before=deadline, rng=rng, device=device)
                assert tuple(sampled[0]) == path
                assert rng.probability == pytest.approx(weight, rel=1e-10, abs=1e-12)


def test_gem_gradient_and_ascent_against_enumeration(monkeypatch):
    hmm = make_hmm(0)
    batch = [[0, 1, 0], [1, 0, 1], [1, 1, 0, 0]]
    csts = constraints('hit_odd')
    data = [np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))]
    for obs in batch:
        ref = enumerate_model(hmm, len(obs), 'hit_odd', 2, obs)[3]
        for dest, source in zip(data, ref):
            dest += source
    params = dna._hmm_to_torch_params(hmm)[:3]
    rng = np.random.default_rng(5)
    logits = [p.log() + torch.tensor(rng.normal(size=p.shape) * .2) for p in params]
    candidate = [p.log_softmax(dim=-1) for p in logits]

    def as_hmm(logp):
        model = copy.deepcopy(hmm)
        p, a, b = [v.exp().numpy() for v in logp]
        model.initprob = dict(enumerate(p))
        model.tprob = {(i, j): a[i, j] for i in range(2) for j in range(2)}
        model.eprob = {(i, j): b[i, j] for i in range(2) for j in range(2)}
        return model

    def reference_surrogate(z):
        logp = [p.log_softmax(dim=-1) for p in z]
        model = as_hmm(logp)
        normalizer = sum(enumerate_model(model, len(obs), 'hit_odd', 2)[2] for obs in batch)
        return sum((p.numpy() * c).sum() for p, c in zip(logp, data)) - normalizer

    prior = [np.zeros_like(c) for c in data]
    for obs in batch:
        for dest, source in zip(prior, enumerate_model(as_hmm(candidate), len(obs), 'hit_odd', 2)[3]):
            dest += source
    gradient = dna._logit_gradient(candidate, [torch.tensor(c) for c in data],
                                  [torch.tensor(c) for c in prior])
    for block, g in enumerate(gradient):
        for index in np.ndindex(g.shape):
            plus, minus = [v.clone() for v in logits], [v.clone() for v in logits]
            plus[block][index] += 1e-6
            minus[block][index] -= 1e-6
            numerical = (reference_surrogate(plus) - reference_surrogate(minus)) / 2e-6
            assert g[index].item() == pytest.approx(numerical, abs=2e-8)

    # Audit the surrogate at successive inner iterates, grouped by frozen E-step counts.
    statistics = dna._constraint_statistics
    iterates = []
    def record(logp, contexts, multiplicities, counts=False):
        result = statistics(logp, contexts, multiplicities, counts)
        if counts:
            iterates.append([p.clone() for p in logp])
        return result
    monkeypatch.setattr(dna, '_constraint_statistics', record)
    fitted, history = dna.generalized_em_constrained(
        hmm, csts, batch, pro_before=2, max_iter=1, inner_max_iter=10, inner_tol=0.)
    iterates.append([p.log() for p in dna._hmm_to_torch_params(fitted)[:3]])
    values = [reference_surrogate(z) for z in iterates]
    assert np.min(np.diff(values)) >= -1e-10
    assert history[-1] > history[0]
    assert len(history) == 2
    assert len(iterates) > 2
    for kind in ['none', 'hit_odd', 'rare']:
        for seed in [0, 2]:
            initial = make_hmm(seed)
            observations = [[0, 0, 0], [0, 0, 0, 0]]
            fitted, history = dna.generalized_em_constrained(
                initial, constraints(kind), observations, pro_before=2, max_iter=5, tol=0.)
            expected = sum(enumerate_model(fitted, len(y), kind, 2, y)[2]
                           - enumerate_model(fitted, len(y), kind, 2)[2] for y in observations)
            assert history[-1] == pytest.approx(expected, abs=1e-10)
            assert np.min(np.diff(history)) >= -1e-10
            assert len(history) == 6
            if seed == 2:
                assert fitted.tprob[0, 0] == 0
                assert fitted.eprob[1, 1] == 0
            assert initial == make_hmm(seed)


    with pytest.warns(RuntimeWarning, match='GEM backtracking failed'):
        fitted, history = dna.generalized_em_constrained(
            hmm, csts, batch, pro_before=2, max_iter=5,
            step_size=1e8, max_backtracks=1)
    np.testing.assert_allclose(history, history[0], atol=1e-12)
    for actual, expected in zip(dna.hmm2numpy(fitted), dna.hmm2numpy(hmm)):
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_dna_notebook_learning_constraints():
    notebook = json.loads(Path(__file__).with_name('dna_clean.ipynb').read_text())
    ns = dict(np=np, torch=torch, copy=copy, itertools=itertools, Munch=Munch)
    for i in [3, 6, 10, 13, 15, 22, 55]:
        exec(''.join(notebook['cells'][i]['source']), ns)
    model, csts = ns['end_hmm'], ns['learning_constraints']
    for i, bad_path in [(0, ['int'] * 10), (3, ['dis']), (3, ['enh']),
                        (1, ['pro', 'int']), (2, ['dis', 'int', 'dis']),
                        (4, ['end', 'pro', 'end'])]:
        states = [r for r in csts[i].m_states if csts[i].init_fun(bad_path[0], r)]
        for k in bad_path[1:]:
            states = [r for r in csts[i].m_states
                      if any(csts[i].update_fun(k, r, prev) for prev in states)]
        assert not any(csts[i].eval_fun(bad_path[-1], r) for r in states)
    samples = dna.sample_constrained_fixed_length(model, csts, 30, 10,
                                                  pro_before=10, rng=np.random.default_rng(23))
    for path, emits in samples:
        assert len(path) == len(emits) == 30
        assert path[-1] == 'end' and 'end' not in path[:-1]
        assert emits[-1] == 'N' and 'N' not in emits[:-1]
        assert 'pro' in path[:10]
        regions = [(k, len(list(group))) for k, group in itertools.groupby(path)]
        assert all(length >= 3 for k, length in regions if k != 'end')
        assert sum(k == 'dis' for k, length in regions) == 1
        assert path.index('pro') < path.index('dis') < path.index('enh')
