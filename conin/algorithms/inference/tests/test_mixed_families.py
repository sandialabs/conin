"""Constrained models whose constraints belong to several families.

A representation that accepts several families applies each family's
constraints in turn: smoek with pyomo for the integer program, smoek with
toulbar2 for toulbar2, and smoek with factor oracles for variable
elimination.  Each mix must give the answer of the same constraints written
in smoek alone.
"""

import pyomo.environ as pyo
import pytest

from conin import (
    ConstrainedGraphicalModel,
    HiddenMarkovModel,
    is_feasible,
    map_query,
    oracle_constraint_fn,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.config import default_mip_solver
from conin.exceptions import InvalidInputError
from conin.util import try_import

with try_import() as smoek_available:
    from conin import smoek_constraint_fn

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

pytestmark = pytest.mark.skipif(not smoek_available, reason="smoek not installed")
needs_mip = pytest.mark.skipif(default_mip_solver is None, reason="no MIP solver")
needs_toulbar2 = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)

READINGS = ["low", "medium", "high", "medium", "low"]


def _hmm():
    activities = ["rest", "walk", "run"]
    transition = {
        "rest": [0.8, 0.15, 0.05],
        "walk": [0.2, 0.7, 0.1],
        "run": [0.1, 0.3, 0.6],
    }
    emission = {
        "rest": {"low": 0.8, "medium": 0.2},
        "walk": {"low": 0.1, "medium": 0.7, "high": 0.2},
        "run": {"medium": 0.4, "high": 0.6},
    }
    return HiddenMarkovModel(
        start_probs={"rest": 0.7, "walk": 0.2, "run": 0.1},
        transition_probs={
            (a, b): p for a in activities for b, p in zip(activities, transition[a])
        },
        emission_probs={(a, o): p for a in activities for o, p in emission[a].items()},
    )


if smoek_available:

    @smoek_constraint_fn()
    def ran(model, data):
        return sum(model.V("H", t, "run") for t in data.T) >= 1

    @smoek_constraint_fn()
    def no_run_at_2_smoek(model, data):
        return model.V("H", 2, "run") == 0


@pyomo_constraint_fn()
def no_run_at_2_pyomo(model, data):
    model.no_run_at_2 = pyo.Constraint(expr=model.V("H", 2, "run") == 0)


@toulbar2_constraint_fn()
def no_run_at_2_toulbar2(model, data):
    model.AddGeneralizedLinearConstraint([model.V("H", 2, "run")], "<=", 0)


@oracle_constraint_fn(nodes=[("H", 2)])
def no_run_at_2_oracle(states):
    return states[("H", 2)] != "run"


def _expected():
    cmodel = ConstrainedGraphicalModel(_hmm(), [ran, no_run_at_2_smoek])
    results = map_query(cmodel, method="variable_elimination", evidence=READINGS)
    return results.solution.states


@pytest.mark.parametrize(
    "other, method, options",
    [
        pytest.param(
            no_run_at_2_pyomo, "integer_program", {}, marks=needs_mip, id="pyomo"
        ),
        pytest.param(
            no_run_at_2_pyomo,
            "integer_program",
            {"ip_formulation": "network_flow"},
            marks=needs_mip,
            id="pyomo-network-flow",
        ),
        pytest.param(
            no_run_at_2_toulbar2, "toulbar2", {}, marks=needs_toulbar2, id="toulbar2"
        ),
        pytest.param(no_run_at_2_oracle, "variable_elimination", {}, id="oracle"),
    ],
)
def test_a_mix_gives_the_answer_of_the_smoek_constraints(other, method, options):
    expected = _expected()
    assert "run" in expected and expected[2] != "run"
    cmodel = ConstrainedGraphicalModel(_hmm(), [ran, other])
    assert cmodel.constraint_families == tuple(sorted({"smoek", type(other).family}))
    results = map_query(cmodel, method=method, evidence=READINGS, **options)
    assert results.solution.states == expected


def test_is_feasible_checks_every_family():
    cmodel = ConstrainedGraphicalModel(_hmm(), [ran, no_run_at_2_oracle])
    assert is_feasible(cmodel, _expected(), observed=READINGS)
    assert not is_feasible(
        cmodel, ["rest", "rest", "run", "rest", "rest"], observed=READINGS
    )
    assert not is_feasible(cmodel, ["rest"] * 5, observed=READINGS)


def test_a_method_without_a_representation_for_the_mix_is_rejected():
    cmodel = ConstrainedGraphicalModel(_hmm(), [ran, no_run_at_2_oracle])
    with pytest.raises(InvalidInputError, match="constraint families"):
        map_query(cmodel, method="integer_program", evidence=READINGS)


def test_families_without_a_common_representation_are_rejected():
    with pytest.raises(InvalidInputError, match="together"):
        ConstrainedGraphicalModel(_hmm(), [no_run_at_2_pyomo, no_run_at_2_toulbar2])
