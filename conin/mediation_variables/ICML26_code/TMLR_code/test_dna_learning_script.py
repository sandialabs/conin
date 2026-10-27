import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from munch import Munch

torch = pytest.importorskip('torch')
import dna_algorithms as dna
import dna_learning as runner


def test_standalone_experiment_matches_notebook_and_plots(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(__file__).parent
    notebook = json.loads((root / 'dna_clean.ipynb').read_text())
    ns = dict(vars(dna), np=np, torch=torch, copy=copy, itertools=itertools,
              Munch=Munch, json=json)
    for index in [3, 6, 10, 13, 15, 22, 55, 56]:
        exec(''.join(notebook['cells'][index]['source']), ns)
    model, constraints = runner.build_learning_model()
    assert model == ns['end_hmm']
    for actual, expected in zip(constraints, ns['learning_constraints']):
        for a, b in zip(dna.create_cst_params(actual, model.states),
                        dna.create_cst_params(expected, model.states)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    settings = dict(num_trials=2, num_samples=2, max_iter=1, inner_max_iter=1, device='cpu')
    output = tmp_path / 'learning_data_gem_v3.json'
    runner.run_experiment(output=output, **settings)
    saved = json.loads(output.read_text())
    assert saved['completed_trials'] == 2
    ns.update(settings)
    exec(''.join(notebook['cells'][57]['source']), ns)
    exec(''.join(notebook['cells'][59]['source']).split('with open(')[0], ns)
    expected = ns['learning_data']
    assert saved['settings'] == expected['settings']
    for prefix in saved['objectives']:
        for metric in ['steps', 'histories', 'terminal_loglik', 'init_l2', 'trans_l2', 'emit_l2']:
            key = f'{prefix}_{metric}'
            np.testing.assert_allclose(saved[key], expected[key], atol=1e-10, rtol=0)

    monkeypatch.chdir(tmp_path)
    plots = json.loads((root.resolve() / 'dna_learning_results.ipynb').read_text())
    plot_namespace = {}
    for cell in plots['cells']:
        if cell['cell_type'] == 'code':
            exec(''.join(cell['source']), plot_namespace)
    for index in plt.get_fignums():
        plt.figure(index).savefig(tmp_path / f'figure-{index}.png')
    plt.close('all')
