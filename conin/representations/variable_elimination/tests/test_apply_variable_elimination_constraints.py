"""Tests for the constraint applier of the variable elimination backend."""

import pytest

from conin.constraints import FactorOracleConstraint, OracleConstraint
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
)
from conin.representations.variable_elimination import (
    apply_variable_elimination_constraints,
)
from conin.representations.variable_elimination.common import factor_model


def _mn():
    return DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        factors=[DiscreteFactor(nodes=["A", "B"], values=[1, 2, 3, 4])],
    )


def _bn():
    return DiscreteBayesianNetwork(
        states={"A": [0, 1]},
        cpds=[DiscreteCPD(node="A", values=[0.4, 0.6])],
    )


@pytest.mark.parametrize("pgm_factory", [_mn, _bn])
def test_no_constraints_gives_no_factors(pgm_factory):
    assert (
        apply_variable_elimination_constraints(pgm=pgm_factory(), constraints=[]) == []
    )


@pytest.mark.parametrize("pgm_factory", [_mn, _bn])
def test_unexpected_constraint_type_message(pgm_factory):
    con = OracleConstraint(func=lambda states: True, name="plain")
    with pytest.raises(TypeError, match="OracleConstraint"):
        apply_variable_elimination_constraints(pgm=pgm_factory(), constraints=[con])


def test_one_factor_per_constraint():
    cons = [
        FactorOracleConstraint(func=lambda s: s["A"] != s["B"], nodes=["A", "B"]),
        FactorOracleConstraint(func=lambda s: s["A"] == 1, nodes=["A"]),
    ]
    factors = apply_variable_elimination_constraints(pgm=_mn(), constraints=cons)
    assert [f.nodes for f in factors] == [["A", "B"], ["A"]]


def test_data_is_passed_to_node_scope_and_predicate():
    seen = []

    def nodes(data):
        seen.append(("nodes", data))
        return ["A"]

    def func(states, data):
        seen.append(("func", data))
        return True

    con = FactorOracleConstraint(func=func, nodes=nodes)
    apply_variable_elimination_constraints(pgm=_bn(), constraints=[con], data="D")
    assert ("nodes", "D") in seen and ("func", "D") in seen


def test_factor_model_of_an_unconstrained_mn_is_the_mn():
    mn = _mn()
    assert factor_model(mn) is mn


def test_factor_model_adds_the_factors_and_keeps_the_input():
    mn = _mn()
    extra = DiscreteFactor(nodes=["A"], values={0: 0, 1: 1})
    model = factor_model(mn, [extra])
    assert isinstance(model, DiscreteMarkovNetwork)
    assert len(model.factors) == 2 and model.factors[-1].nodes == ["A"]
    assert len(mn.factors) == 1


def test_factor_model_of_a_bn_has_one_factor_per_cpd():
    model = factor_model(_bn())
    assert isinstance(model, DiscreteMarkovNetwork)
    assert [f.nodes for f in model.factors] == [["A"]]
    assert dict(model.factors[0].values) == {0: 0.4, 1: 0.6}


def test_factor_model_rejects_other_models():
    with pytest.raises(TypeError, match="Unexpected model type"):
        factor_model(object())
