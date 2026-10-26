"""Tests for the Pyomo solution parsers.

They are tested without a solver: the variable values of a built model are set
by hand, and the or_topas (aos) solutions are stubbed from the same values.
The or_topas paths are also run end to end with a stubbed ``aos`` module, so
the shape of their results (``states`` on every solution) is checked; the
network-flow parser used to return ``variable_value`` instead of ``states``.
"""

import importlib
import types

import munch
import pyomo.environ as pyo
import pytest

import conin.examples.hidden_markov_model as hex_
import conin.examples.markov_network as mex
from conin.constrained_models import (
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.exceptions import InsufficientSolutionsError
from conin.inference import map_query
from conin.inference.pyomo import ip_inference, solve_pyomo_map_query_model
from conin.inference.pyomo.solve import (
    parse_aos_solution_pyomo_map_query,
    parse_model_solution_pyomo_map_query,
)
from conin.inference.pyomo.hmm_network_flow import (
    parse_aos_solution_pyomo_ip_inference,
    parse_model_solution_pyomo_ip_inference,
)

solve_module = importlib.import_module("conin.inference.pyomo.solve")
network_flow_module = importlib.import_module("conin.inference.pyomo.hmm_network_flow")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


class _AosVar:
    def __init__(self, var):
        self.name = var.name
        self.value = var.value
        self.fixed = var.is_fixed()


class _AosSolution:
    """The parts of an or_topas solution that the parsers read."""

    def __init__(self, model, objective):
        self._variables = [_AosVar(v) for v in model.component_data_objects(pyo.Var)]
        self._by_name = {v.name: v for v in self._variables}
        self._objectives = [types.SimpleNamespace(value=objective)]

    def variable(self, name):
        return self._by_name[name]


def _set_mn_assignment(model, assignment):
    """Set model.x and model.y to the 0/1 values of a total assignment."""
    for (r, s), var in model.V.items():
        if not var.is_fixed():
            var.set_value(1 if assignment[r] == s.value else 0)
    for var in model.y.values():
        var.set_value(0)


def _abc_model(evidence=None):
    repn = ConstrainedDiscreteMarkovNetwork(mex.ABC_conin().pgm).representation("pyomo")
    return repn.create_model(evidence=evidence)


def _network_flow_model(observed):
    chmm = ConstrainedHiddenMarkovModel(hex_.create_hmm1())
    repn = chmm.representation("pyomo_network_flow")
    M, data = repn.create_model(observed=observed)
    return repn.hidden_markov_model, M, data


def _set_hidden_path(M, data, hmm, hidden):
    for (t, a), var in M.hmm.x.items():
        var.set_value(1 if hidden[t] == hmm.hidden_to_external[a] else 0)


# ---------------------------------------------------------------------
# parse_*_solution_pyomo_map_query
# ---------------------------------------------------------------------

ABC = {"A": 2, "B": 2, "C": 1}


def test_parse_model_solution_reads_the_assignment():
    model = _abc_model()
    _set_mn_assignment(model, ABC)
    soln = parse_model_solution_pyomo_map_query(model, False)
    assert soln.states == ABC
    assert set(soln) == {"states", "log_factor_sum"}


@pytest.mark.parametrize("with_fixed", [False, True])
def test_parse_model_solution_with_fixed_evidence(with_fixed):
    model = _abc_model(evidence={"B": 2})
    _set_mn_assignment(model, ABC)
    soln = parse_model_solution_pyomo_map_query(model, with_fixed)
    expected = dict(ABC) if with_fixed else {"A": 2, "C": 1}
    assert soln.states == expected


def test_parse_model_solution_missing_value_raises():
    model = _abc_model()
    for var in model.x.values():
        var.set_value(0)
    with pytest.raises(AssertionError, match="do not have values"):
        parse_model_solution_pyomo_map_query(model, False)


@pytest.mark.parametrize("with_fixed", [False, True])
def test_parse_aos_solution_matches_parse_model_solution(with_fixed):
    model = _abc_model(evidence={"B": 2})
    _set_mn_assignment(model, ABC)
    expected = parse_model_solution_pyomo_map_query(model, with_fixed)
    aos_solution = _AosSolution(model, objective=expected.log_factor_sum)
    assert parse_aos_solution_pyomo_map_query(model, aos_solution, with_fixed) == (
        expected
    )


# ---------------------------------------------------------------------
# parse_*_solution_pyomo_ip_inference (network flow)
# ---------------------------------------------------------------------

HIDDEN = ["h0", "h1", "h1"]


def test_parse_network_flow_model_solution():
    hmm, M, data = _network_flow_model(["o0", "o1", "o1"])
    _set_hidden_path(M, data, hmm, HIDDEN)
    for var in M.component_data_objects(pyo.Var):
        if var.value is None:
            var.set_value(0)
    soln = parse_model_solution_pyomo_ip_inference(
        M=M, hmm=hmm, T=3, log_likelihood=-1.5, data=data
    )
    assert soln.states == HIDDEN
    assert soln.hidden == HIDDEN
    assert soln.log_likelihood == -1.5
    assert set(soln) == {"states", "hidden", "log_likelihood", "variables"}


def test_parse_network_flow_aos_solution_has_states():
    hmm, M, data = _network_flow_model(["o0", "o1", "o1"])
    _set_hidden_path(M, data, hmm, HIDDEN)
    for var in M.component_data_objects(pyo.Var):
        if var.value is None:
            var.set_value(0)
    soln = parse_aos_solution_pyomo_ip_inference(
        aos_solution=_AosSolution(M, objective=-2.0), M=M, hmm=hmm, T=3, data=data
    )
    assert soln.states == HIDDEN
    assert soln.hidden == HIDDEN
    assert soln.log_likelihood == -2.0
    assert set(soln) == {"states", "hidden", "log_likelihood", "variables"}


def test_parse_network_flow_solution_missing_state_raises():
    hmm, M, data = _network_flow_model(["o0", "o1"])
    for var in M.component_data_objects(pyo.Var):
        var.set_value(0)
    with pytest.raises(AssertionError, match="missing hidden state"):
        parse_model_solution_pyomo_ip_inference(
            M=M, hmm=hmm, T=2, log_likelihood=0.0, data=data
        )


# ---------------------------------------------------------------------
# The or_topas paths end to end (aos stubbed)
# ---------------------------------------------------------------------


def _stub_aos(solutions_for):
    """An ``aos`` module whose methods return ``solutions_for(model)``."""

    def run(model, **options):
        return types.SimpleNamespace(solutions=solutions_for(model))

    return types.SimpleNamespace(
        enumerate_binary_solutions=run, gurobi_generate_solutions=run
    )


def _use_aos(monkeypatch, module, solutions_for):
    monkeypatch.setattr(module, "or_topas_available", True)
    monkeypatch.setattr(module, "aos", _stub_aos(solutions_for), raising=False)


def _network_flow_solutions(paths):
    """aos solutions of a network-flow model, one per hidden path."""

    def solutions_for(M):
        hmm = hex_.create_hmm1()
        result = []
        for i, hidden in enumerate(paths):
            for (t, a), var in M.hmm.x.items():
                var.set_value(1 if hidden[t] == hmm.hidden_to_external[a] else 0)
            for var in M.component_data_objects(pyo.Var):
                if var.value is None:
                    var.set_value(0)
            result.append(_AosSolution(M, objective=-float(i)))
        return result

    return solutions_for


PATHS = [["h0", "h0", "h1"], ["h1", "h1", "h1"]]


def test_ip_inference_or_topas_solutions(monkeypatch):
    _use_aos(monkeypatch, network_flow_module, _network_flow_solutions(PATHS))
    chmm = hex_.create_chmm1_pyomo()
    results = ip_inference(hmm=chmm, observed=["o0", "o1", "o1"], solver="or_topas")
    assert [s.states for s in results.solutions] == PATHS
    assert results.solution is results.solutions[0]
    assert [s.log_likelihood for s in results.solutions] == [0.0, -1.0]


def test_map_query_network_flow_or_topas_with_dict_evidence(monkeypatch):
    # key_solutions_by_time reads soln.states, which the or_topas solutions
    # did not have.
    _use_aos(monkeypatch, network_flow_module, _network_flow_solutions(PATHS))
    results = map_query(
        hex_.create_chmm1_pyomo(),
        method="integer_program",
        ip_formulation="network_flow",
        evidence={0: "o0", 1: "o1", 2: "o1"},
        solver="or_topas",
    )
    assert [s.states for s in results.solutions] == [dict(enumerate(p)) for p in PATHS]
    assert results.solution.states == {0: "h0", 1: "h0", 2: "h1"}


def test_solve_pyomo_map_query_model_or_topas_solutions(monkeypatch):
    assignments = [ABC, {"A": 0, "B": 0, "C": 0}]

    def solutions_for(model):
        result = []
        for i, assignment in enumerate(assignments):
            _set_mn_assignment(model, assignment)
            result.append(_AosSolution(model, objective=-float(i)))
        return result

    _use_aos(monkeypatch, solve_module, solutions_for)
    results = solve_pyomo_map_query_model(_abc_model(), solver="or_topas")
    assert [s.states for s in results.solutions] == assignments
    assert results.solution is results.solutions[0]
    assert [s.log_factor_sum for s in results.solutions] == [0.0, -1.0]
    assert results.termination_condition == "ok"


@pytest.mark.parametrize(
    "module,solve",
    [
        (
            solve_module,
            lambda: solve_pyomo_map_query_model(_abc_model(), solver="or_topas"),
        ),
        (
            network_flow_module,
            lambda: ip_inference(
                hmm=hex_.create_chmm1_pyomo(), observed=["o0"], solver="or_topas"
            ),
        ),
    ],
    ids=["solve_pyomo_map_query_model", "ip_inference"],
)
def test_or_topas_without_solutions_raises(monkeypatch, module, solve):
    _use_aos(monkeypatch, module, lambda model: [])
    with pytest.raises(InsufficientSolutionsError, match="No solutions"):
        solve()


def test_unknown_topas_method_raises(monkeypatch):
    _use_aos(monkeypatch, network_flow_module, lambda model: [])
    with pytest.raises(RuntimeError, match="not supported"):
        ip_inference(
            hmm=hex_.create_chmm1_pyomo(),
            observed=["o0"],
            solver="or_topas",
            solver_options=munch.Munch(topas_method="nope"),
        )


def test_lp_inference_finds_one_solution():
    from conin.exceptions import InvalidInputError
    from conin.inference.pyomo import lp_inference

    with pytest.raises(InvalidInputError, match="num_solutions must be 1, got 2"):
        lp_inference(hmm=hex_.create_hmm1(), observed=["o0"], num_solutions=2)
