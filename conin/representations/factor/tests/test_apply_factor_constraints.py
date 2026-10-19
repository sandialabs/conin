"""Regression tests for the constraint helpers in VariableElimination.py."""

import pytest

from conin.constraints import FactorOracleConstraint, OracleConstraint
from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
)
from conin.inference.VariableElimination import (
    _add_bn_constraints_as_evidence,
    _add_mn_constraints_as_evidence,
)


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


@pytest.mark.parametrize(
    "helper, pgm_factory",
    [
        (_add_mn_constraints_as_evidence, _mn),
        (_add_bn_constraints_as_evidence, _bn),
    ],
)
def test_no_constraints_returns_pgm(helper, pgm_factory):
    pgm = pgm_factory()
    evidence = {}
    assert helper(pgm, [], None, evidence) is pgm
    assert evidence == {}


@pytest.mark.parametrize(
    "helper, pgm_factory",
    [
        (_add_mn_constraints_as_evidence, _mn),
        (_add_bn_constraints_as_evidence, _bn),
    ],
)
def test_unexpected_constraint_type_message(helper, pgm_factory):
    con = OracleConstraint(func=lambda states: True, name="plain")
    with pytest.raises(TypeError, match="OracleConstraint"):
        helper(pgm_factory(), [con], None, {})


def test_factor_oracle_still_accepted():
    con = FactorOracleConstraint(func=lambda s: s["A"] != s["B"], nodes=["A", "B"])
    evidence = {}
    new_pgm = _add_mn_constraints_as_evidence(_mn(), [con], None, evidence)
    assert new_pgm is not None
    assert list(evidence.values()) == [1]
