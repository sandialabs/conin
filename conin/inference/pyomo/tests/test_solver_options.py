"""The or_topas paths must not change the caller's ``solver_options``.

They used to ``pop("topas_method")`` from the caller's dict, so a second call
with the same dict silently used the default ``"balas"`` method. or_topas is an
optional dependency, so these tests replace it with a stub that records the
options it gets and then stops the solve.
"""

import importlib
import types

import pytest

import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.constrained_models import (
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.inference.pyomo import ip_inference, solve_pyomo_map_query_model

solve_module = importlib.import_module("conin.inference.pyomo.solve")
network_flow_module = importlib.import_module("conin.inference.pyomo.hmm_network_flow")


class _Stop(Exception):
    pass


def _stub_aos(calls):
    def record(method):
        def run(model, **options):
            calls.append((method, options))
            raise _Stop()

        return run

    return types.SimpleNamespace(
        enumerate_binary_solutions=record("balas"),
        gurobi_generate_solutions=record("gurobi_solution_pool"),
    )


def _solve_mn(solver_options):
    model = (
        ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
        .representation("pyomo")
        .create_model()
    )
    solve_pyomo_map_query_model(model, solver="or_topas", solver_options=solver_options)


def _solve_network_flow(solver_options):
    chmm = hex_.create_chmm1_pyomo()
    ip_inference(
        hmm=chmm,
        observed=["o0"] * 5,
        solver="or_topas",
        solver_options=solver_options,
    )


@pytest.mark.parametrize(
    "module,solve",
    [(solve_module, _solve_mn), (network_flow_module, _solve_network_flow)],
    ids=["solve_pyomo_map_query_model", "ip_inference"],
)
def test_or_topas_leaves_solver_options_unchanged(monkeypatch, module, solve):
    calls = []
    monkeypatch.setattr(module, "or_topas_available", True)
    monkeypatch.setattr(module, "aos", _stub_aos(calls), raising=False)

    solver_options = {"topas_method": "gurobi_solution_pool", "num_solutions": 3}
    for _ in range(2):
        with pytest.raises(_Stop):
            solve(solver_options)

    assert solver_options == {
        "topas_method": "gurobi_solution_pool",
        "num_solutions": 3,
    }
    # Both calls used the requested method and got the other options.
    assert [method for method, _ in calls] == ["gurobi_solution_pool"] * 2
    assert all(options.get("num_solutions") == 3 for _, options in calls)
    assert all("topas_method" not in options for _, options in calls)


@pytest.mark.parametrize(
    "module,solve",
    [(solve_module, _solve_mn), (network_flow_module, _solve_network_flow)],
    ids=["solve_pyomo_map_query_model", "ip_inference"],
)
def test_or_topas_without_solver_options_uses_balas(monkeypatch, module, solve):
    calls = []
    monkeypatch.setattr(module, "or_topas_available", True)
    monkeypatch.setattr(module, "aos", _stub_aos(calls), raising=False)
    with pytest.raises(_Stop):
        solve(None)
    assert [method for method, _ in calls] == ["balas"]
