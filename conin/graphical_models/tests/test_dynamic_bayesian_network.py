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


# ---------------------------------------------------------------------
# The accessors (states_of, card, nodes, edges) and the argument checks
# ---------------------------------------------------------------------


def _mixed_dbn():
    """A DBN with a static node S and a dynamic node Z."""
    t = ExpressionVariable()
    return DynamicDiscreteBayesianNetwork(
        states={"S": ["lo", "mid", "hi"]},
        dynamic_states={"Z": [0, 1]},
        cpds=[DiscreteCPD(node="S", values=[0.2, 0.3, 0.5])] + _z_cpds(t),
        t=t,
    )


@pytest.mark.parametrize(
    "node,states",
    [("S", ["lo", "mid", "hi"]), ("Z", [0, 1]), (("Z", 0), [0, 1]), (("Z", 7), [0, 1])],
)
def test_states_of_and_card(node, states):
    dbn = _mixed_dbn()
    assert dbn.states_of(node) == states
    assert dbn.card(node) == len(states)


@pytest.mark.parametrize("node", ["X", ("X", 0), ("S", 0)])
def test_states_of_and_card_unknown_node(node):
    dbn = _mixed_dbn()
    with pytest.raises(ValueError, match="Unexpected node value"):
        dbn.states_of(node)
    with pytest.raises(ValueError, match="Unexpected node value"):
        dbn.card(node)


def test_nodes_states_and_edges():
    dbn = _mixed_dbn()
    assert dbn.nodes == ["S"]
    assert dbn.dynamic_nodes == ["Z"]
    assert dbn.states == {"S": ["lo", "mid", "hi"]}
    assert dbn.dynamic_states == {"Z": [0, 1]}
    assert dbn.edges == [] and dbn.dynamic_edges == []
    assert dbn.check_model() is True


def test_states_and_dynamic_states_are_copies():
    dbn = _mixed_dbn()
    dbn.states["S"] = ["x"]
    dbn.dynamic_states["Z"] = ["x"]
    assert dbn.states_of("S") == ["lo", "mid", "hi"]
    assert dbn.states_of("Z") == [0, 1]


def test_list_states_name_the_nodes_by_index():
    dbn = DynamicDiscreteBayesianNetwork(
        states=[3, 2], cpds=[DiscreteCPD(node=0, values=[0.2, 0.3, 0.5])]
    )
    assert dbn.nodes == [0, 1]
    assert dbn.states == {0: [0, 1, 2], 1: [0, 1]}
    assert dbn.card(1) == 2


def test_states_must_be_a_list_or_dict():
    with pytest.raises(TypeError, match="Unexpected type for states"):
        DynamicDiscreteBayesianNetwork(
            states=("A",), cpds=[DiscreteCPD(node="A", values=[1.0])]
        )


def test_dynamic_states_must_be_a_dict():
    with pytest.raises(TypeError, match="Unexpected type for dynamic states"):
        DynamicDiscreteBayesianNetwork(
            dynamic_states=[("Z", [0, 1])],
            cpds=[DiscreteCPD(node=("Z", 0), values=[0.5, 0.5])],
        )
