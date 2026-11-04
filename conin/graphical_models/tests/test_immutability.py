"""The graphical models cannot be changed through their accessors or their inputs.

Before, ``mn.nodes.append(...)``, ``mn.states_of(node).append(...)`` and
``mn.factors[0].values[key] = ...`` (and the same for Bayesian networks and
DBNs) silently changed the model.
"""

import copy
import pickle

import pytest

from conin.graphical_models import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    HiddenMarkovModel,
    TimeIndex,
)


def _mn():
    return DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        factors=[
            DiscreteFactor(
                nodes=["A", "B"],
                values={(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0},
            )
        ],
    )


def _bn():
    return DiscreteBayesianNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        cpds=[
            DiscreteCPD(node="A", values=[0.4, 0.6]),
            DiscreteCPD(node="B", parents=["A"], values={0: [0.2, 0.8], 1: [0.9, 0.1]}),
        ],
    )


def _dbn():
    t = TimeIndex()
    return DynamicDiscreteBayesianNetwork(
        dynamic_states={"X": [0, 1]},
        t=t,
        cpds=[
            DiscreteCPD(node=("X", 0), values=[0.5, 0.5]),
            DiscreteCPD(
                node=("X", t),
                parents=[("X", t - 1)],
                values={0: [0.7, 0.3], 1: [0.4, 0.6]},
            ),
        ],
    )


def _hmm():
    return HiddenMarkovModel(
        start_probs={"h0": 0.4, "h1": 0.6},
        transition_probs={("h0", "h0"): 0.9, ("h0", "h1"): 0.1, ("h1", "h1"): 1.0},
        emission_probs={("h0", "o0"): 1.0, ("h1", "o1"): 1.0},
    )


def _raises(mutate):
    with pytest.raises(TypeError, match="read-only"):
        mutate()


# The accessors that used to leak, per model: each must raise and leave the
# model as it was.
MUTATIONS = {
    "mn": [
        lambda m: m.nodes.append("Z"),
        lambda m: m.states_of("A").append(9),
        lambda m: m.states["A"].append(9),
        lambda m: m.edges.append(("A", "Z")),
        lambda m: m.factors[0].values.__setitem__((0, 0), 99.0),
        lambda m: m.factors[0].nodes.append("Z"),
    ],
    "bn": [
        lambda m: m.nodes.append("Z"),
        lambda m: m.states_of("A").append(9),
        lambda m: m.states["A"].append(9),
        lambda m: m.edges.append(("B", "A")),
        lambda m: m.cpds[0].values.__setitem__(0, 0.99),
        lambda m: m.cpds[1].values[0].__setitem__(0, 0.99),
        lambda m: m.cpds[1].parents.append("Z"),
    ],
    "dbn": [
        lambda m: m.dynamic_nodes.append("Z"),
        lambda m: m.states_of("X").append(9),
        lambda m: m.dynamic_states["X"].append(9),
        lambda m: m.dynamic_edges.append(("X", "X")),
        lambda m: m.cpds[1].values[0].__setitem__(0, 0.99),
    ],
    "hmm": [
        lambda m: m.hidden_states.append("h2"),
        lambda m: m.observable_states.__setitem__(0, "o9"),
        lambda m: m.start_probs.__setitem__("h0", 1.0),
        lambda m: m.transition_probs.pop(("h0", "h0")),
        lambda m: m.emission_probs.update({("h0", "o0"): 0.5}),
    ],
}
FACTORIES = {"mn": _mn, "bn": _bn, "dbn": _dbn, "hmm": _hmm}


def _snapshot(model):
    if isinstance(model, HiddenMarkovModel):
        return [
            repr(model.hidden_states),
            repr(model.observable_states),
            repr(model.start_probs),
            repr(model.transition_probs),
            repr(model.emission_probs),
        ]
    parts = [repr(model.nodes), repr(model.states), repr(model.edges)]
    # The values, not the reprs: a DBN's CPD nodes hold its time index, whose
    # repr is its address (a copied DBN has its own).
    if isinstance(model, DiscreteMarkovNetwork):
        parts.append([(f.nodes, f.values) for f in model.factors])
    else:
        parts.append([cpd.values for cpd in model.cpds])
    if isinstance(model, DynamicDiscreteBayesianNetwork):
        parts += [repr(model.dynamic_nodes), repr(model.dynamic_states)]
    return parts


@pytest.mark.parametrize(
    "kind,i",
    [(kind, i) for kind, ms in MUTATIONS.items() for i in range(len(ms))],
)
def test_accessors_are_read_only(kind, i):
    model = FACTORIES[kind]()
    before = _snapshot(model)
    _raises(lambda: MUTATIONS[kind][i](model))
    assert _snapshot(model) == before


def test_returned_lists_and_dicts_still_compare_and_print_as_before():
    mn = _mn()
    assert mn.nodes == ["A", "B"] and repr(mn.nodes) == "['A', 'B']"
    assert mn.states_of("A") == [0, 1]
    assert mn.states == {"A": [0, 1], "B": [0, 1]}
    bn = _bn()
    assert bn.cpds[1].values == {0: {0: 0.2, 1: 0.8}, 1: {0: 0.9, 1: 0.1}}
    assert bn.cpds[1].parents == ["A"]


def test_the_states_and_factor_lists_are_new_copies():
    # Their containers may be changed by the caller; the model is not.
    mn = _mn()
    states = mn.states
    states["Z"] = [0]
    factors = mn.factors
    factors.clear()
    assert "Z" not in mn.states and len(mn.factors) == 1


def test_changing_the_inputs_does_not_change_the_model():
    states = {"A": [0, 1], "B": [0, 1]}
    values = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
    factor_nodes = ["A", "B"]
    mn = DiscreteMarkovNetwork(
        states=states, factors=[DiscreteFactor(nodes=factor_nodes, values=values)]
    )
    states["A"].append(2)
    values[(0, 0)] = 99.0
    factor_nodes.append("Z")
    assert mn.states_of("A") == [0, 1]
    assert mn.factors[0].values[(0, 0)] == 1.0
    assert mn.factors[0].nodes == ["A", "B"]

    table = {0: {0: 0.2, 1: 0.8}, 1: {0: 0.9, 1: 0.1}}
    parents = ["A"]
    bn = DiscreteBayesianNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        cpds=[
            DiscreteCPD(node="A", values=[0.4, 0.6]),
            DiscreteCPD(node="B", parents=parents, values=table),
        ],
    )
    table[0][0] = 0.99
    parents.append("Z")
    assert bn.cpds[1].values[0][0] == 0.2
    assert bn.cpds[1].parents == ["A"]


def test_mn_edges_are_fixed_at_construction():
    # The inferred edges used to be cached on first access and then passed to
    # with_factor as if they had been given, so the new factor's edges were
    # missing.
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        factors=_mn().factors,
    )
    assert mn.edges == [("A", "B")]
    mn3 = mn.with_factor(
        DiscreteFactor(
            nodes=["B", "C"], values={(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 2}
        )
    )
    assert sorted(mn3.edges) == [("A", "B"), ("B", "C")]


def test_dbn_time_index_cannot_be_replaced():
    dbn = _dbn()
    with pytest.raises(AttributeError):
        dbn.t = TimeIndex()


def test_pickled_and_copied_hmms_stay_read_only_and_equal():
    hmm = _hmm()
    for other in (pickle.loads(pickle.dumps(hmm)), copy.deepcopy(hmm)):
        assert other == hmm and _snapshot(other) == _snapshot(hmm)
        _raises(lambda: other.hidden_states.append("h2"))
        _raises(lambda: other.emission_probs.clear())


@pytest.mark.parametrize("kind", ["mn", "bn", "dbn"])
def test_pickled_and_copied_models_stay_read_only(kind):
    model = FACTORIES[kind]()
    for other in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        assert _snapshot(other) == _snapshot(model)
        cpds_or_factors = other.factors if kind == "mn" else other.cpds
        _raises(lambda: cpds_or_factors[0].values.clear())
        node = "X" if kind == "dbn" else "A"
        _raises(lambda: other.states_of(node).append(9))
