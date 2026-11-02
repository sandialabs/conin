import itertools

import pytest

from conin.constraints import FactorOracleConstraint, OracleConstraint
from conin.graphical_models import (
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DiscreteCPD,
    DiscreteBayesianNetwork,
)
from conin.representations.variable_elimination import materialise_factor_oracle


def _mn():
    return DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1], "C": [0, 1]},
        factors=[
            DiscreteFactor(nodes=["A", "B"], values=[1, 2, 3, 4]),
            DiscreteFactor(nodes=["B", "C"], values=[1, 2, 3, 4]),
        ],
    )


def _bn():
    return DiscreteBayesianNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        cpds=[
            DiscreteCPD(node="A", values=[0.4, 0.6]),
            DiscreteCPD(
                node="B",
                parents=["A"],
                values={0: {0: 0.3, 1: 0.7}, 1: {0: 0.9, 1: 0.1}},
            ),
        ],
    )


def test_mn_factor():
    con = FactorOracleConstraint(
        func=lambda s: s["A"] != s["C"], nodes=["A", "C"], name="diff"
    )
    factor = materialise_factor_oracle(con, _mn())
    assert isinstance(factor, DiscreteFactor)
    assert factor.nodes == ["A", "C"]
    assert factor.values == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}


def test_mn_factor_callable_nodes_with_data():
    def nodes(data):
        yield from data["scope"]

    con = FactorOracleConstraint(
        func=lambda s, data: sum(s.values()) == data["total"], nodes=nodes
    )
    data = {"scope": ["A", "B", "C"], "total": 2}
    factor = materialise_factor_oracle(con, _mn(), data)
    assert factor.nodes == ["A", "B", "C"]
    expected = {
        states: (1 if sum(states) == 2 else 0)
        for states in itertools.product([0, 1], repeat=3)
    }
    assert factor.values == expected


def test_bn_factor():
    """A Bayesian network gets the same indicator factor (no constraint node)."""
    con = FactorOracleConstraint(
        func=lambda s: s["A"] == 1 or s["B"] == 1, nodes=["A", "B"], name="either"
    )
    factor = materialise_factor_oracle(con, _bn())
    assert isinstance(factor, DiscreteFactor)
    assert factor.nodes == ["A", "B"]
    assert factor.values == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}


def test_factor_over_one_node_is_keyed_by_states():
    con = FactorOracleConstraint(func=lambda s: s["A"] == 1, nodes=["A"], name="a1")
    factor = materialise_factor_oracle(con, _mn())
    assert factor.nodes == ["A"]
    assert factor.values == {0: 0, 1: 1}


def test_callable_nodes_with_data_over_one_node():
    def nodes(data):
        yield data

    con = FactorOracleConstraint(
        func=lambda s, data: s[data] == 0, nodes=nodes, name="zero"
    )
    factor = materialise_factor_oracle(con, _bn(), "B")
    assert factor.nodes == ["B"]
    assert factor.values == {0: 1, 1: 0}


def test_plain_oracle_raises():
    con = OracleConstraint(func=lambda s: True)
    with pytest.raises(TypeError):
        materialise_factor_oracle(con, _mn())
