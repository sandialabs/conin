"""Tests for DiscreteFactor, DiscreteMarkovNetwork, and ConstrainedDiscreteMarkovNetwork."""

import pytest

from conin.markov_network import (
    ConstrainedDiscreteMarkovNetwork,
    DiscreteFactor,
    DiscreteMarkovNetwork,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_mn():
    """Two-node MN with one pairwise factor and two unary factors."""
    mn = DiscreteMarkovNetwork()
    mn.states = {"A": [0, 1], "B": [0, 1]}
    f_a = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
    f_b = DiscreteFactor(nodes=["B"], values={0: 3.0, 1: 1.0})
    f_ab = DiscreteFactor(
        nodes=["A", "B"], values={(0, 0): 1.0, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 2.0}
    )
    mn.factors = [f_a, f_b, f_ab]
    mn.check_model()
    return mn


# ---------------------------------------------------------------------------
# DiscreteFactor tests
# ---------------------------------------------------------------------------


class TestDiscreteFactor:
    def test_nodes_attribute(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        assert f.nodes == ["A"]

    def test_values_attribute(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        assert f.values == {0: 1.0, 1: 2.0}

    def test_default_value_default(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0})
        assert f.default_value == 0

    def test_default_value_custom(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0}, default_value=99)
        assert f.default_value == 99

    # --- assignments ---

    def test_assignments_unary(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1, 2]}
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0, 2: 3.0})
        result = list(f.assignments(mn.states))
        # One entry per state, each is a list of one (node, value) pair
        assert result == [[("A", 0)], [("A", 1)], [("A", 2)]]

    def test_assignments_binary(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        f = DiscreteFactor(
            nodes=["A", "B"],
            values={(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0},
        )
        result = list(f.assignments(mn.states))
        assert len(result) == 4  # 2 × 2 assignments
        assert [("A", 0), ("B", 0)] in result
        assert [("A", 1), ("B", 1)] in result

    # --- normalize ---

    def test_normalize_already_dict_returns_self(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        assert f.normalize(mn) is f

    def test_normalize_list_unary(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": ["x", "y"]}
        f = DiscreteFactor(nodes=["A"], values=[3.0, 7.0])
        normalized = f.normalize(mn)
        assert normalized.values == {"x": 3.0, "y": 7.0}

    def test_normalize_list_binary(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        # List order follows the Cartesian product in the order nodes are listed
        f = DiscreteFactor(nodes=["A", "B"], values=[1.0, 2.0, 3.0, 4.0])
        normalized = f.normalize(mn)
        assert isinstance(normalized.values, dict)
        assert len(normalized.values) == 4


# ---------------------------------------------------------------------------
# DiscreteMarkovNetwork tests
# ---------------------------------------------------------------------------


class TestDiscreteMarkovNetwork:

    # --- states setter ---

    def test_states_setter_dict(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        assert set(mn.nodes) == {"A", "B"}
        assert mn.states_of("A") == [0, 1]

    def test_states_setter_list(self):
        mn = DiscreteMarkovNetwork()
        mn.states = [3, 2]
        assert mn.nodes == [0, 1]
        assert mn.states_of(0) == [0, 1, 2]
        assert mn.states_of(1) == [0, 1]

    def test_states_setter_invalid_type(self):
        mn = DiscreteMarkovNetwork()
        with pytest.raises(TypeError):
            mn.states = "bad"

    # --- states_of / card ---

    def test_states_of(self):
        mn = _make_simple_mn()
        assert mn.states_of("A") == [0, 1]
        assert mn.states_of("B") == [0, 1]

    def test_card(self):
        mn = _make_simple_mn()
        assert mn.card("A") == 2
        assert mn.card("B") == 2

    # --- edges ---

    def test_edges_inferred_from_factors(self):
        mn = _make_simple_mn()
        # edges is inferred lazily from the pairwise factor scope
        assert ("A", "B") in mn.edges

    def test_edges_explicit_setter(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        mn.edges = [("A", "B")]
        assert mn.edges == [("A", "B")]

    def test_edges_unary_factor_has_no_edges(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        mn.factors = [DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})]
        assert mn.edges == []

    # --- factors setter normalizes list values ---

    def test_factors_setter_normalizes_list(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        mn.factors = [DiscreteFactor(nodes=["A"], values=[1.0, 2.0])]
        # After setting, list values should be normalized to a dict
        assert isinstance(mn.factors[0].values, dict)

    def test_factors_setter_keeps_dict(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        mn.factors = [f]
        assert mn.factors[0] is f  # already a dict, returned unchanged

    # --- add_factor ---

    def test_add_factor_appends(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        mn.add_factor(f)
        assert f in mn.factors

    def test_add_factor_registers_new_node(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        # Start with only A in the factors; add_factor with B should register it
        f = DiscreteFactor(nodes=["B"], values={0: 1.0, 1: 2.0})
        mn.add_factor(f)
        assert "B" in mn.nodes

    # --- num_factor_parameters ---

    def test_num_factor_parameters(self):
        mn = _make_simple_mn()
        # f_a: 2, f_b: 2, f_ab: 4
        assert mn.num_factor_parameters() == 8

    # --- check_model ---

    def test_check_model_passes(self):
        _make_simple_mn()  # should not raise

    def test_check_model_negative_value(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        mn.factors = [DiscreteFactor(nodes=["A"], values={0: -1.0, 1: 2.0})]
        with pytest.raises(AssertionError):
            mn.check_model()

    def test_check_model_invalid_node_state_in_factor(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        # State 99 does not exist for node A
        mn.factors = [DiscreteFactor(nodes=["A"], values={99: 1.0, 0: 1.0})]
        with pytest.raises(AssertionError):
            mn.check_model()

    def test_check_model_factor_node_not_in_model(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        # Factor references node B which is not in states
        mn._factors = [DiscreteFactor(nodes=["B"], values={0: 1.0, 1: 2.0})]
        with pytest.raises((AssertionError, KeyError)):
            mn.check_model()

    def test_check_model_edge_node_not_in_model(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        mn._edges = [("A", "Z")]  # Z is not in states
        mn._factors = [DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})]
        with pytest.raises(RuntimeError):
            mn.check_model()

    def test_check_model_list_factor_wrong_length(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1]}
        # Should have 2 values for a binary node, not 3
        mn._factors = [DiscreteFactor(nodes=["A"], values=[1.0, 2.0, 3.0])]
        with pytest.raises(AssertionError):
            mn.check_model()

    def test_check_model_missing_node(self):
        mn = DiscreteMarkovNetwork()
        mn.states = {"A": [0, 1], "B": [0, 1]}
        # Only A has a factor; B is missing
        mn._factors = [DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})]
        with pytest.raises(AssertionError):
            mn.check_model()


# ---------------------------------------------------------------------------
# ConstrainedDiscreteMarkovNetwork tests
# ---------------------------------------------------------------------------


class TestConstrainedDiscreteMarkovNetwork:
    def test_wraps_pgm(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        assert cmn.pgm is mn

    def test_nodes_delegates_to_pgm(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        assert cmn.nodes == mn.nodes

    def test_states_of_delegates(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        assert cmn.states_of("A") == [0, 1]

    def test_default_constraints_empty(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        assert cmn.constraints == []

    def test_constraints_kwarg(self):
        mn = _make_simple_mn()
        sentinel = object()
        cmn = ConstrainedDiscreteMarkovNetwork(mn, constraints=[sentinel])
        assert cmn.constraints == [sentinel]

    def test_constraints_setter(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        sentinel = object()
        cmn.constraints = [sentinel]
        assert cmn.constraints == [sentinel]

    def test_constraints_setter_requires_list(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        with pytest.raises(AssertionError):
            cmn.constraints = "not-a-list"

    def test_check_model_delegates(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        cmn.check_model()  # should not raise
