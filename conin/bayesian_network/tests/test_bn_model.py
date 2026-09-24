"""Tests for DiscreteCPD, DiscreteBayesianNetwork, and create_mn_from_bn."""

import pytest

from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD
from conin.bayesian_network.bn_to_mn import create_mn_from_bn
from conin.markov_network import DiscreteMarkovNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_bn():
    """Two-node BN: A -> B."""
    bn = DiscreteBayesianNetwork()
    bn.states = {"A": [0, 1], "B": [0, 1]}
    cpd_A = DiscreteCPD(node="A", values=[0.9, 0.1])
    cpd_B = DiscreteCPD(
        node="B",
        parents=["A"],
        values={0: [0.2, 0.8], 1: [0.9, 0.1]},
    )
    bn.cpds = [cpd_A, cpd_B]
    bn.check_model()
    return bn


# ---------------------------------------------------------------------------
# DiscreteCPD tests
# ---------------------------------------------------------------------------


class TestDiscreteCPD:
    def test_node_attribute(self):
        cpd = DiscreteCPD(node="X", values=[0.5, 0.5])
        assert cpd.node == "X"

    def test_values_attribute(self):
        cpd = DiscreteCPD(node="X", values=[0.3, 0.7])
        assert cpd.values == [0.3, 0.7]

    def test_parents_default_is_none(self):
        cpd = DiscreteCPD(node="X", values=[0.5, 0.5])
        assert cpd.parents is None

    def test_str(self):
        cpd = DiscreteCPD(node="X", values=[0.5, 0.5])
        s = str(cpd)
        assert "X" in s
        assert "0.5" in s

    def test_normalize_list_values_no_parents(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": ["a0", "a1"]}
        cpd = DiscreteCPD(node="A", values=[0.3, 0.7])
        normalized = cpd.normalize(bn)
        # A list CPD without parents normalizes to a dict keyed by state values.
        assert normalized.values == {"a0": 0.3, "a1": 0.7}

    def test_normalize_dict_list_values_with_parent(self):
        bn = _make_simple_bn()
        # Retrieve the already-normalized CPD from the network.
        cpd_B = bn.cpds[1]
        # After normalization through cpds setter the inner values are dicts.
        assert isinstance(next(iter(cpd_B.values.values())), dict)

    def test_normalize_already_dict_of_dicts_is_identity(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": [0, 1], "B": [0, 1]}
        cpd_B = DiscreteCPD(
            node="B",
            parents=["A"],
            values={0: {0: 0.2, 1: 0.8}, 1: {0: 0.9, 1: 0.1}},
        )
        normalized = cpd_B.normalize(bn)
        # Already fully normalized — should be returned unchanged.
        assert normalized is cpd_B

    def test_to_factor_no_parents(self):
        cpd = DiscreteCPD(node="A", values=[0.9, 0.1])
        factor = cpd.to_factor()
        assert factor.nodes == ["A"]
        assert factor.values == [0.9, 0.1]

    def test_to_factor_with_parents(self):
        cpd = DiscreteCPD(
            node="B",
            parents=["A"],
            values={0: {0: 0.2, 1: 0.8}, 1: {0: 0.9, 1: 0.1}},
        )
        factor = cpd.to_factor()
        assert factor.nodes == ["A", "B"]
        # Values should be keyed by (parent_val, node_val) tuples.
        assert factor.values[(0, 0)] == 0.2
        assert factor.values[(0, 1)] == 0.8
        assert factor.values[(1, 0)] == 0.9
        assert factor.values[(1, 1)] == 0.1


# ---------------------------------------------------------------------------
# DiscreteBayesianNetwork tests
# ---------------------------------------------------------------------------


class TestDiscreteBayesianNetwork:
    def test_states_setter_dict(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": [0, 1], "B": [0, 1]}
        assert set(bn.nodes) == {"A", "B"}

    def test_states_setter_list(self):
        bn = DiscreteBayesianNetwork()
        bn.states = [2, 3]
        # Nodes are integer indices when states is a list.
        assert bn.nodes == [0, 1]
        assert bn.states_of(0) == [0, 1]
        assert bn.states_of(1) == [0, 1, 2]

    def test_states_setter_invalid_type(self):
        bn = DiscreteBayesianNetwork()
        with pytest.raises(TypeError):
            bn.states = "bad"

    def test_states_of(self):
        bn = _make_simple_bn()
        assert bn.states_of("A") == [0, 1]
        assert bn.states_of("B") == [0, 1]

    def test_card(self):
        bn = _make_simple_bn()
        assert bn.card("A") == 2
        assert bn.card("B") == 2

    def test_edges(self):
        bn = _make_simple_bn()
        assert ("A", "B") in bn.edges

    def test_edges_no_parents(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": [0, 1]}
        cpd = DiscreteCPD(node="A", values=[0.5, 0.5])
        bn.cpds = [cpd]
        bn.check_model()
        assert bn.edges == []

    def test_num_cpd_parameters(self):
        bn = _make_simple_bn()
        # CPD A: 2 parameters (one per state). CPD B: 2 parent states × 1 key each.
        assert bn.num_cpd_parameters() == 2 + 2

    def test_check_model_passes(self):
        # Should not raise.
        _make_simple_bn()

    def test_check_model_unknown_node_in_cpd(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": [0, 1]}
        cpd_B = DiscreteCPD(node="B", values=[0.5, 0.5])
        bn._cpds = [cpd_B]
        with pytest.raises(AssertionError):
            bn.check_model()

    def test_check_model_missing_node(self):
        bn = DiscreteBayesianNetwork()
        bn.states = {"A": [0, 1], "B": [0, 1]}
        # Only provide CPD for A, leaving B without one.
        bn._cpds = [DiscreteCPD(node="A", values=[0.5, 0.5])]
        with pytest.raises(AssertionError):
            bn.check_model()


# ---------------------------------------------------------------------------
# create_mn_from_bn tests
# ---------------------------------------------------------------------------


class TestCreateMNFromBN:
    def test_returns_discrete_markov_network(self):
        bn = _make_simple_bn()
        mn = create_mn_from_bn(bn)
        assert isinstance(mn, DiscreteMarkovNetwork)

    def test_states_preserved(self):
        bn = _make_simple_bn()
        mn = create_mn_from_bn(bn)
        assert mn.states == bn.states

    def test_number_of_factors(self):
        bn = _make_simple_bn()
        mn = create_mn_from_bn(bn)
        # One factor per CPD.
        assert len(mn.factors) == len(bn.cpds)

    def test_factor_nodes(self):
        bn = _make_simple_bn()
        mn = create_mn_from_bn(bn)
        factor_node_sets = [set(f.nodes) for f in mn.factors]
        # Factor for CPD A should include only A; factor for CPD B should include A and B.
        assert {"A"} in factor_node_sets
        assert {"A", "B"} in factor_node_sets
