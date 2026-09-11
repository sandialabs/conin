"""Run the three-method DNA learning experiment without the figure notebook."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from munch import Munch

from dna_algorithms import (
    baum_welch_joint,
    baum_welch_unconstrained,
    generalized_em_constrained,
    hmm2numpy,
    randomize_hmm_like,
    sample_constrained_fixed_length,
)


DEFAULT_OUTPUT = Path(__file__).resolve().with_name('learning_data_gem_v3.json')


def build_learning_model():
    """Return the notebook's ground-truth HMM and corrected learning constraints."""
    states = ['pro', 'ex1', 'ex2', 'int', 'dis', 'enh', 'end']
    emits = ['A', 'T', 'C', 'G', 'N']
    initial = [.5, 0, 0, .5, 0, 0, 0]
    transition = [
        [.6, .1, .1, .2, 0, 0, 0],
        [0, .4, .3, .2, .1, 0, 0],
        [0, .3, .4, 0, .2, .1, 0],
        [.1, .1, .1, .4, 0, .1, .2],
        [0, 1/3, 1/3, 0, 1/3, 0, 0],
        [0, .25, .25, .25, 0, .25, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    emission = [
        [.1, .1, .4, .4, 0],
        [.1, .1, .7, .1, 0],
        [.7, .1, .1, .1, 0],
        [.25, .25, .25, .25, 0],
        [.5, 0, .5, 0, 0],
        [.4, .4, .1, .1, 0],
        [0, 0, 0, 0, 1],
    ]
    hmm = Munch(
        states=states, emits=emits,
        initprob=dict(zip(states, initial)),
        tprob={(s, t): transition[i][j] for i, s in enumerate(states)
               for j, t in enumerate(states)},
        eprob={(s, e): emission[i][j] for i, s in enumerate(states)
               for j, e in enumerate(emits)},
    )
    promoter = Munch(
        m_states=[True, False],
        init_fun=lambda k, r: r == (k == 'pro'),
        update_fun=lambda k, r, prev: r == (prev or k == 'pro'),
        eval_fun=lambda k, r: bool(r),
    )
    stay = Munch(
        m_states=list(itertools.product(states, range(1, 4))),
        init_fun=lambda k, r: r == (k, 1),
        update_fun=lambda k, r, prev: (
            (prev[1] == 3 or k == prev[0])
            and r == (k, min(prev[1] + 1, 3) if k == prev[0] else 1)),
        eval_fun=lambda k, r: True,
    )
    disease = Munch(
        m_states=list(itertools.product([True, False], range(3))),
        init_fun=lambda k, r: r == (k != 'dis', int(k == 'dis')),
        update_fun=lambda k, r, prev: r == (
            k != 'dis', min(prev[1] + int(prev[0] and k == 'dis'), 2)),
        eval_fun=lambda k, r: r[1] == 1,
    )
    precedence = Munch(
        m_states=list(itertools.product([True, False], repeat=3)),
        init_fun=lambda k, r: k not in ('dis', 'enh') and r == (k == 'pro', False, False),
        update_fun=lambda k, r, prev: (
            (k != 'dis' or prev[0]) and (k != 'enh' or prev[1])
            and r == (prev[0] or k == 'pro', prev[1] or k == 'dis', prev[2] or k == 'enh')),
        eval_fun=lambda k, r: r[2],
    )
    end = Munch(
        m_states=[0, 1, 2],
        init_fun=lambda k, r: r == int(k == 'end'),
        update_fun=lambda k, r, prev: r == min(prev + int(k == 'end'), 2),
        eval_fun=lambda k, r: k == 'end' and r < 2,
    )
    return hmm, [promoter, stay, disease, precedence, end]


def run_experiment(
    *, output=DEFAULT_OUTPUT, seed=23, num_trials=20, num_samples=30,
    time_horizon=30, pro_before=10, max_iter=500, tol=1e-5,
    inner_max_iter=10, inner_tol=1e-6, step_size=1., max_backtracks=30,
    device='auto', verbose=False,
):
    """Fit all three methods and atomically save results after each completed trial.

    The JSON uses the same result keys as dna_clean. Tolerance measures total
    batch improvement; the inner GEM tolerance uses the mean surrogate.
    """
    if num_trials < 1 or num_samples < 1:
        raise ValueError('num_trials and num_samples must be positive')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=seed, num_trials=num_trials, num_samples=num_samples,
                    time_horizon=time_horizon, pro_before=pro_before,
                    max_iter=max_iter, tol=tol, inner_max_iter=inner_max_iter,
                    inner_tol=inner_tol, step_size=step_size,
                    max_backtracks=max_backtracks, dtype='float64', device=device)
    data = dict(
        experiment_version='fixed_horizon_gem_v3_joint_comparison',
        settings=settings, completed_trials=0,
        objectives={'constrained': 'sum log p(y | C_T)',
                    'joint': 'sum log p(y, C_T)', 'unconstrained': 'sum log p(y)'},
    )
    for prefix in data['objectives']:
        for metric in ['steps', 'terminal_loglik', 'histories', 'init_l2', 'trans_l2', 'emit_l2']:
            data[f'{prefix}_{metric}'] = []
    hmm, constraints = build_learning_model()
    truth = hmm2numpy(hmm)
    rng = np.random.default_rng(seed)
    methods = [
        ('constrained', generalized_em_constrained, dict(
            cst_list=constraints, pro_before=pro_before,
            inner_max_iter=inner_max_iter, inner_tol=inner_tol,
            step_size=step_size, max_backtracks=max_backtracks)),
        ('unconstrained', baum_welch_unconstrained, dict(pseudocount=0.)),
        ('joint', baum_welch_joint, dict(cst_list=constraints, pro_before=pro_before)),
    ]
    print(f'Running on {device}; saving to {output}', flush=True)
    for trial in range(num_trials):
        initial = randomize_hmm_like(hmm, rng=rng, alpha=1.)
        samples = sample_constrained_fixed_length(
            hmm, constraints, time_horizon, num_samples,
            pro_before=pro_before, rng=rng, device=device)
        observations = [y for _, y in samples]
        for prefix, fit, options in methods:
            print(f'Trial {trial + 1}/{num_trials}: {prefix}', flush=True)
            fitted, history = fit(
                initial, obs_batch=observations, max_iter=max_iter, tol=tol,
                dtype=torch.float64, device=device, verbose=verbose, **options)
            data[f'{prefix}_histories'].append(history)
            data[f'{prefix}_steps'].append(len(history) - 1)
            data[f'{prefix}_terminal_loglik'].append(history[-1])
            for metric, actual, expected in zip(
                    ['init_l2', 'trans_l2', 'emit_l2'], hmm2numpy(fitted), truth):
                data[f'{prefix}_{metric}'].append(float(np.linalg.norm(actual - expected)))
        data['completed_trials'] = trial + 1
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        temporary.replace(output)
        print(f'Saved {trial + 1} completed trials', flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='JSON destination (default: next to this script)')
    for name, default in [('seed', 23), ('num-trials', 20), ('num-samples', 30),
                          ('time-horizon', 30), ('pro-before', 10), ('max-iter', 500),
                          ('inner-max-iter', 10), ('max-backtracks', 30)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--tol', type=float, default=1e-5,
                        help='total batch log-likelihood improvement threshold')
    parser.add_argument('--inner-tol', type=float, default=1e-6)
    parser.add_argument('--step-size', type=float, default=1.)
    parser.add_argument('--device', default='auto', help='auto, cpu, or a CUDA device such as cuda:0')
    parser.add_argument('--verbose', action='store_true', help='print per-iteration likelihoods')
    run_experiment(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
