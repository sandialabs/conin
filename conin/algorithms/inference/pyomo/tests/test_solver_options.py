"""The ``solver_options`` and ``tee`` reach the solvers.

The or_topas paths must not change the caller's ``solver_options``: they used
to ``pop("topas_method")`` from the caller's dict, so a second call with the
same dict silently used the default ``"balas"`` method. or_topas is an
optional dependency, so these tests replace it with a stub that records the
options it gets and then stops the solve.  Both or_topas paths go through
:func:`~conin.algorithms.inference.pyomo.solve.solve_with_or_topas`, so the
stub replaces or_topas in its module.
"""

import importlib
import types

import pyomo.environ as pyo

import pytest

import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.algorithms.inference.pyomo import (
    ip_inference,
    lp_inference,
    solve_pyomo_map_query_model,
)

solve_module = importlib.import_module("conin.algorithms.inference.pyomo.solve")
network_flow_module = importlib.import_module(
    "conin.algorithms.inference.pyomo.hmm_network_flow"
)


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


def _solve_mn(solver_options, tee=False):
    model = (
        ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm)
        .representation("pyomo")
        .create_model()
    )
    solve_pyomo_map_query_model(
        model, solver="or_topas", solver_options=solver_options, tee=tee
    )


def _solve_network_flow(solver_options, tee=False):
    chmm = hex_.create_chmm1_pyomo()
    ip_inference(
        hmm=chmm,
        observed=["o0"] * 5,
        solver="or_topas",
        solver_options=solver_options,
        quiet=not tee,
    )


SOLVES = pytest.mark.parametrize(
    "solve",
    [_solve_mn, _solve_network_flow],
    ids=["solve_pyomo_map_query_model", "ip_inference"],
)


def _stub_or_topas(monkeypatch, calls):
    monkeypatch.setattr(solve_module, "or_topas_available", True)
    monkeypatch.setattr(solve_module, "aos", _stub_aos(calls), raising=False)


@SOLVES
def test_or_topas_leaves_solver_options_unchanged(monkeypatch, solve):
    calls = []
    _stub_or_topas(monkeypatch, calls)

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


@SOLVES
def test_or_topas_without_solver_options_uses_balas(monkeypatch, solve):
    calls = []
    _stub_or_topas(monkeypatch, calls)
    with pytest.raises(_Stop):
        solve(None)
    assert [method for method, _ in calls] == ["balas"]


@SOLVES
@pytest.mark.parametrize("topas_method", ["balas", "gurobi_solution_pool"])
@pytest.mark.parametrize("tee", [False, True])
def test_or_topas_gets_tee(monkeypatch, solve, topas_method, tee):
    # The network-flow path never passed tee, and balas never got it.
    calls = []
    _stub_or_topas(monkeypatch, calls)
    with pytest.raises(_Stop):
        solve({"topas_method": topas_method}, tee=tee)
    assert calls == [(topas_method, {"tee": tee})]


@pytest.mark.skipif(
    not pyo.SolverFactory("glpk").available(exception_flag=False),
    reason="glpk not installed",
)
@pytest.mark.parametrize(
    "infer,make",
    [(lp_inference, hex_.create_hmm1), (ip_inference, hex_.create_chmm1_pyomo)],
    ids=["lp_inference", "ip_inference"],
)
def test_network_flow_solver_options_reach_the_solver(monkeypatch, infer, make):
    # They were ignored for every solver except or_topas.
    solvers = []
    factory = pyo.SolverFactory

    def recording_factory(name):
        opt = factory(name)
        solvers.append(opt)
        return opt

    monkeypatch.setattr(network_flow_module.pyo, "SolverFactory", recording_factory)
    results = infer(
        hmm=make(),
        observed=["o0"] * 12,
        solver="glpk",
        solver_options={"tmlim": 100},
    )
    assert results.solution is not None
    assert [dict(opt.options) for opt in solvers] == [{"tmlim": 100}]
