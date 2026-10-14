"""Tests for the DynamicDiscreteBayesianNetwork constructor."""

import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteCPD,
    DynamicDiscreteBayesianNetwork,
    ExpressionVariable,
)


def _z_cpds(t):
    return [
        DiscreteCPD(node=("Z", 0), values=[0.5, 0.5]),
        DiscreteCPD(
            node=("Z", t),
            parents=[("Z", t - 1)],
            values={0: [0.7, 0.3], 1: [0.8, 0.2]},
        ),
    ]


def test_constructor_uses_the_given_time_variable():
    t = ExpressionVariable()
    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states={"Z": [0, 1]}, cpds=_z_cpds(t), t=t
    )
    assert dbn.t is t
    assert dbn.cpds[1].node == ("Z", t)


def test_cpds_required():
    with pytest.raises(TypeError):
        DynamicDiscreteBayesianNetwork()
    with pytest.raises(TypeError):
        DynamicDiscreteBayesianNetwork(dynamic_states={"Z": [0, 1]})


def test_cpds_must_be_non_empty():
    with pytest.raises(InvalidInputError, match="cpds"):
        DynamicDiscreteBayesianNetwork(dynamic_states={"Z": [0, 1]}, cpds=[])


def test_some_states_required():
    t = ExpressionVariable()
    with pytest.raises(InvalidInputError, match="states"):
        DynamicDiscreteBayesianNetwork(cpds=_z_cpds(t), t=t)
    with pytest.raises(InvalidInputError, match="states"):
        DynamicDiscreteBayesianNetwork(
            states={}, dynamic_states={}, cpds=_z_cpds(t), t=t
        )


def test_static_states_alone_are_enough():
    dbn = DynamicDiscreteBayesianNetwork(
        states={"X": ["T", "F"]},
        cpds=[DiscreteCPD(node="X", values=[0.9, 0.1])],
    )
    assert dbn.nodes == ["X"]
    assert dbn.dynamic_nodes == []


def test_with_cpd_does_not_mutate():
    t = ExpressionVariable()
    cpds = _z_cpds(t)
    dbn = DynamicDiscreteBayesianNetwork(
        dynamic_states={"Z": [0, 1]}, cpds=cpds[:1], t=t
    )
    dbn2 = dbn.with_cpd(cpds[1])
    assert len(dbn.cpds) == 1
    assert len(dbn2.cpds) == 2
    assert dbn2.t is t
