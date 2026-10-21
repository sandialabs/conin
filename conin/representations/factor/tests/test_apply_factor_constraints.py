"""Tests for the factor constraint applier, :func:`apply_factor_constraints`.

The first two groups are regression tests for an empty constraint list and for
the ``TypeError`` message.
"""

import pytest

from conin.constraints import FactorOracleConstraint, OracleConstraint
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
)
from conin.representations.factor import apply_factor_constraints


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
def test_no_constraints_returns_pgm(pgm_factory):
    pgm = pgm_factory()
    new_pgm, evidence = apply_factor_constraints(pgm=pgm, constraints=[])
    assert new_pgm is pgm
    assert evidence == {}


@pytest.mark.parametrize("pgm_factory", [_mn, _bn])
def test_unexpected_constraint_type_message(pgm_factory):
    con = OracleConstraint(func=lambda states: True, name="plain")
    with pytest.raises(TypeError, match="OracleConstraint"):
        apply_factor_constraints(pgm=pgm_factory(), constraints=[con])


def test_unexpected_model_type():
    con = FactorOracleConstraint(func=lambda s: True, nodes=["A"])
    with pytest.raises(TypeError, match="Unexpected model type"):
        apply_factor_constraints(pgm=object(), constraints=[con])


def test_mn_factor_oracle_adds_a_factor_and_no_evidence():
    """Regression: the MN path used to force the last scope node to 1."""
    pgm = _mn()
    con = FactorOracleConstraint(func=lambda s: s["A"] != s["B"], nodes=["A", "B"])
    new_pgm, evidence = apply_factor_constraints(pgm=pgm, constraints=[con])
    assert evidence == {}
    assert len(new_pgm.factors) == len(pgm.factors) + 1
    assert new_pgm.factors[-1].nodes == ["A", "B"]
    assert len(pgm.factors) == 1  # the input model is unchanged


def test_bn_factor_oracle_adds_a_cpd_node_and_its_evidence():
    pgm = _bn()
    con = FactorOracleConstraint(func=lambda s: s["A"] == 1, nodes=["A"], name="c")
    new_pgm, evidence = apply_factor_constraints(pgm=pgm, constraints=[con])
    assert evidence == {("c", -1): 1}
    assert ("c", -1) in new_pgm.nodes
    assert ("c", -1) not in pgm.nodes


def test_data_is_passed_to_node_scope_and_predicate():
    seen = []

    def nodes(data):
        seen.append(("nodes", data))
        return ["A"]

    def func(states, data):
        seen.append(("func", data))
        return True

    con = FactorOracleConstraint(func=func, nodes=nodes)
    apply_factor_constraints(pgm=_bn(), constraints=[con], data="D")
    assert ("nodes", "D") in seen and ("func", "D") in seen
