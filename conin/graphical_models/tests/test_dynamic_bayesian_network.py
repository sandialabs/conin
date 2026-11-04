"""Tests for the DynamicDiscreteBayesianNetwork constructor."""

import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models import (
    DiscreteCPD,
    DynamicDiscreteBayesianNetwork,
    TimeIndex,
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
    t = TimeIndex()
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
    t = TimeIndex()
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


# ---------------------------------------------------------------------
# The accessors (states_of, card, nodes, edges) and the argument checks
# ---------------------------------------------------------------------


def _mixed_dbn():
    """A DBN with a static node S and a dynamic node Z."""
    t = TimeIndex()
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
    with pytest.raises(KeyError):
        dbn.states_of(node)
    with pytest.raises(KeyError):
        dbn.card(node)


def test_nodes_states_and_edges():
    dbn = _mixed_dbn()
    assert dbn.nodes == ["S"]
    assert dbn.dynamic_nodes == ["Z"]
    assert dbn.states == {"S": ["lo", "mid", "hi"]}
    assert dbn.dynamic_states == {"Z": [0, 1]}
    assert dbn.edges == [] and dbn.dynamic_edges == []
    dbn._check_model()


def test_states_and_dynamic_states_are_copies():
    dbn = _mixed_dbn()
    dbn.states["S"] = ["x"]
    dbn.dynamic_states["Z"] = ["x"]
    assert dbn.states_of("S") == ["lo", "mid", "hi"]
    assert dbn.states_of("Z") == [0, 1]


def test_list_states_name_the_nodes_by_index():
    dbn = DynamicDiscreteBayesianNetwork(
        states=[3, 2],
        cpds=[
            DiscreteCPD(node=0, values=[0.2, 0.3, 0.5]),
            DiscreteCPD(node=1, values=[0.4, 0.6]),
        ],
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


def _ab_dbn(t, cpds):
    return DynamicDiscreteBayesianNetwork(
        dynamic_states={"A": [0, 1], "B": [0, 1]}, cpds=cpds, t=t
    )


def _a_cpds(t):
    return [
        DiscreteCPD(node=("A", 0), values=[0.5, 0.5]),
        DiscreteCPD(
            node=("A", t), parents=[("A", t - 1)], values={0: [0.2, 0.8], 1: [0.9, 0.1]}
        ),
    ]


def _b_given_a(node, parent):
    return DiscreteCPD(
        node=node, parents=[parent], values={0: [0.2, 0.8], 1: [0.9, 0.1]}
    )


def test_check_model_accepts_a_valid_dbn():
    t = TimeIndex()
    dbn = _ab_dbn(t, _a_cpds(t) + [_b_given_a(("B", t), ("A", t))])
    dbn._check_model()
    # The time variable is left as it was.
    assert t.value() is None


# The constructor validates the network.


def test_initial_cpd_where_the_template_applies():
    # The template B_t | A_t also applies at time 0.
    t = TimeIndex()
    with pytest.raises(
        InvalidInputError,
        match=r"Invalid dynamic Bayesian network \(unrolled over 0..2\): "
        r"There is more than one CPD for node \('B', 0\)",
    ):
        _ab_dbn(
            t,
            _a_cpds(t)
            + [_b_given_a(("B", 0), ("A", 0)), _b_given_a(("B", t), ("A", t))],
        )


def test_missing_initial_cpd():
    # A_t | A_{t-1} applies from time 1: A_0 needs its own CPD.
    t = TimeIndex()
    with pytest.raises(InvalidInputError, match=r"no CPD for the nodes \[\('A', 0\)\]"):
        _ab_dbn(t, _a_cpds(t)[1:] + [_b_given_a(("B", t), ("A", t))])


def test_template_distribution():
    t = TimeIndex()
    bad = DiscreteCPD(
        node=("A", t), parents=[("A", t - 1)], values={0: [0.2, 0.9], 1: [0.9, 0.1]}
    )
    with pytest.raises(InvalidInputError, match=r"\('A', 1\) .* must sum to 1"):
        _ab_dbn(t, [_a_cpds(t)[0], bad, _b_given_a(("B", t), ("A", t))])


def test_unrolls_past_the_longest_lag():
    # B_t | B_{t-2} applies from time 2, so B_0 and B_1 need initial CPDs.
    t = TimeIndex()
    b_t = _b_given_a(("B", t), ("B", t - 2))
    b0 = DiscreteCPD(node=("B", 0), values=[0.5, 0.5])
    b1 = DiscreteCPD(node=("B", 1), values=[0.5, 0.5])
    _ab_dbn(t, _a_cpds(t) + [b0, b1, b_t])
    with pytest.raises(
        InvalidInputError, match=r"0..3\): There is no CPD for the nodes \[\('B', 1\)\]"
    ):
        _ab_dbn(t, _a_cpds(t) + [b0, b_t])


@pytest.mark.parametrize(
    "extra,match",
    [
        # W is not a dynamic node
        (lambda t: DiscreteCPD(node=("W", t), values={0: 0.5, 1: 0.5}), r"\('W', "),
        # X is not a static node
        (lambda t: DiscreteCPD(node="X", values={0: 0.5, 1: 0.5}), "'X'"),
        # nor is the parent Y
        (
            lambda t: DiscreteCPD(
                node=("B", t), parents=["Y"], values={0: [0.5, 0.5], 1: [0.5, 0.5]}
            ),
            "'Y'",
        ),
    ],
)
def test_cpds_must_use_the_nodes_of_the_network(extra, match):
    # Before, these were accepted, and unrolling made up nodes such as
    # ("W", <TimeIndex>).
    t = TimeIndex()
    with pytest.raises(InvalidInputError, match=match + ".*neither a static node"):
        _ab_dbn(t, _a_cpds(t) + [_b_given_a(("B", t), ("A", t)), extra(t)])


@pytest.mark.parametrize(
    "time", [None, "t", TimeIndex()], ids=["none", "str", "other_t"]
)
def test_cpd_times_must_use_the_time_index(time):
    # ("A", None) used to fail with AttributeError while checking; a CPD
    # written against another TimeIndex cannot be unrolled.
    t = TimeIndex()
    with pytest.raises(InvalidInputError, match=r"The CPD of \('A', "):
        _ab_dbn(
            t,
            _a_cpds(t)[:1]
            + [
                DiscreteCPD(node=("A", time), values=[0.5, 0.5]),
                _b_given_a(("B", t), ("A", t)),
            ],
        )
