"""Tests for DiscreteFactor and DiscreteMarkovNetwork."""

import pytest

from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_mn():
    """Two-node MN with one pairwise factor and two unary factors."""
    f_a = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
    f_b = DiscreteFactor(nodes=["B"], values={0: 3.0, 1: 1.0})
    f_ab = DiscreteFactor(
        nodes=["A", "B"], values={(0, 0): 1.0, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 2.0}
    )
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]}, factors=[f_a, f_b, f_ab]
    )
    mn.check_model()
    return mn


def _unit_factors(states):
    """Return one unary factor with unit weights per node of ``states``."""
    if isinstance(states, list):
        states = {i: list(range(card)) for i, card in enumerate(states)}
    return [
        DiscreteFactor(nodes=[node], values={v: 1.0 for v in values})
        for node, values in states.items()
    ]


def _mn(states):
    """Return a Markov network over ``states`` with unit unary factors."""
    return DiscreteMarkovNetwork(states=states, factors=_unit_factors(states))


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
        mn = _mn({"A": [0, 1, 2]})
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0, 2: 3.0})
        result = list(f.assignments(mn.states))
        # One entry per state, each is a list of one (node, value) pair
        assert result == [[("A", 0)], [("A", 1)], [("A", 2)]]

    def test_assignments_binary(self):
        mn = _mn({"A": [0, 1], "B": [0, 1]})
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
        mn = _mn({"A": [0, 1]})
        assert f.normalize(mn) is f

    def test_normalize_list_unary(self):
        mn = _mn({"A": ["x", "y"]})
        f = DiscreteFactor(nodes=["A"], values=[3.0, 7.0])
        normalized = f.normalize(mn)
        assert normalized.values == {"x": 3.0, "y": 7.0}

    def test_normalize_list_binary(self):
        mn = _mn({"A": [0, 1], "B": [0, 1]})
        # List order follows the Cartesian product in the order nodes are listed
        f = DiscreteFactor(nodes=["A", "B"], values=[1.0, 2.0, 3.0, 4.0])
        normalized = f.normalize(mn)
        assert isinstance(normalized.values, dict)
        assert len(normalized.values) == 4


# ---------------------------------------------------------------------------
# DiscreteMarkovNetwork tests
# ---------------------------------------------------------------------------


class TestDiscreteMarkovNetwork:

    # --- states constructor argument ---

    def test_states_dict(self):
        mn = _mn({"A": [0, 1], "B": [0, 1]})
        assert set(mn.nodes) == {"A", "B"}
        assert mn.states_of("A") == [0, 1]

    def test_states_list(self):
        mn = _mn([3, 2])
        assert mn.nodes == [0, 1]
        assert mn.states_of(0) == [0, 1, 2]
        assert mn.states_of(1) == [0, 1]

    def test_states_invalid_type(self):
        with pytest.raises(TypeError):
            DiscreteMarkovNetwork(states="bad", factors=_unit_factors({"A": [0, 1]}))

    def test_factors_required(self):
        with pytest.raises(TypeError):
            DiscreteMarkovNetwork()
        with pytest.raises(TypeError):
            DiscreteMarkovNetwork(states={"A": [0, 1]})

    def test_factors_must_be_non_empty(self):
        with pytest.raises(InvalidInputError, match="factors"):
            DiscreteMarkovNetwork(states={"A": [0, 1]}, factors=[])

    def test_nodes_inferred_from_factors(self):
        mn = DiscreteMarkovNetwork(
            factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})]
        )
        assert mn.nodes == ["A"]

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

    def test_edges_explicit_constructor_arg(self):
        states = {"A": [0, 1], "B": [0, 1]}
        mn = DiscreteMarkovNetwork(
            states=states, edges=[("A", "B")], factors=_unit_factors(states)
        )
        assert mn.edges == [("A", "B")]

    def test_edges_unary_factor_has_no_edges(self):
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})],
        )
        assert mn.edges == []

    # --- factors constructor argument normalizes list values ---

    def test_factors_normalizes_list(self):
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values=[1.0, 2.0])],
        )
        # After construction, list values should be normalized to a dict
        assert isinstance(mn.factors[0].values, dict)

    def test_factors_keeps_dict(self):
        f = DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})
        mn = DiscreteMarkovNetwork(states={"A": [0, 1]}, factors=[f])
        assert mn.factors[0] is f  # already a dict, returned unchanged

    # --- with_factor ---

    def test_with_factor_appends(self):
        mn = _make_simple_mn()
        f = DiscreteFactor(nodes=["A"], values={0: 5.0, 1: 7.0})
        mn2 = mn.with_factor(f)
        assert f in mn2.factors
        # with_factor does not mutate the original model
        assert f not in mn.factors

    def test_with_factor_registers_new_node(self):
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})],
        )
        # B is in neither the states nor the factors; with_factor should register it
        f = DiscreteFactor(nodes=["B"], values={0: 1.0, 1: 2.0})
        mn2 = mn.with_factor(f)
        assert "B" in mn2.nodes

    # --- num_factor_parameters ---

    def test_num_factor_parameters(self):
        mn = _make_simple_mn()
        # f_a: 2, f_b: 2, f_ab: 4
        assert mn.num_factor_parameters() == 8

    # --- check_model ---

    def test_check_model_passes(self):
        _make_simple_mn()  # should not raise

    def test_check_model_negative_value(self):
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={0: -1.0, 1: 2.0})],
        )
        with pytest.raises(InvalidInputError):
            mn.check_model()

    def test_check_model_invalid_node_state_in_factor(self):
        # State 99 does not exist for node A
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={99: 1.0, 0: 1.0})],
        )
        with pytest.raises(InvalidInputError):
            mn.check_model()

    def test_check_model_factor_node_not_in_model(self):
        # Factor references node B which is not in states
        mn = _mn({"A": [0, 1]})
        mn._factors = [DiscreteFactor(nodes=["B"], values={0: 1.0, 1: 2.0})]
        with pytest.raises(InvalidInputError):
            mn.check_model()

    def test_check_model_edge_node_not_in_model(self):
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})],
        )
        mn._edges = [("A", "Z")]  # Z is not in states
        with pytest.raises(InvalidInputError):
            mn.check_model()

    def test_check_model_list_factor_wrong_length(self):
        # Should have 2 values for a binary node, not 3
        mn = _mn({"A": [0, 1]})
        mn._factors = [DiscreteFactor(nodes=["A"], values=[1.0, 2.0, 3.0])]
        with pytest.raises(InvalidInputError):
            mn.check_model()

    def test_check_model_missing_node(self):
        # Only A has a factor; B is missing
        mn = DiscreteMarkovNetwork(
            states={"A": [0, 1], "B": [0, 1]},
            factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})],
        )
        with pytest.raises(InvalidInputError):
            mn.check_model()


# ---------------------------------------------------------------------
# check_model raises InvalidInputError with a message (it used assert, so the
# checks vanished under python -O)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "factors,match",
    [
        ([DiscreteFactor(nodes=["A"], values={0: -1.0, 1: 2.0})], "negative"),
        ([DiscreteFactor(nodes=["A"], values=[1.0, -2.0])], "negative"),
        (
            [DiscreteFactor(nodes=["A"], values={99: 1.0, 0: 1.0})],
            "Unexpected node value",
        ),
        (
            [DiscreteFactor(nodes=["A"], values=[1.0, 2.0, 3.0])],
            "has 3 values, expected 2",
        ),
        ([DiscreteFactor(nodes=["B"], values={0: 1.0, 1: 2.0})], "Unexpected node 'B'"),
    ],
)
def test_check_model_messages(factors, match):
    mn = _mn({"A": [0, 1]})
    mn._factors = factors
    with pytest.raises(InvalidInputError, match=match):
        mn.check_model()


def test_check_model_names_the_nodes_without_a_factor():
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1]},
        factors=[DiscreteFactor(nodes=["A"], values={0: 1.0, 1: 2.0})],
    )
    with pytest.raises(InvalidInputError, match=r"without a factor: \['B'\]"):
        mn.check_model()


def test_check_model_list_factor_over_a_subset_of_the_nodes():
    # The expected length is the product of the cardinalities of the factor's
    # own nodes.
    mn = DiscreteMarkovNetwork(
        states={"A": [0, 1], "B": [0, 1, 2]},
        factors=[
            DiscreteFactor(nodes=["A"], values=[1.0, 2.0]),
            DiscreteFactor(nodes=["A", "B"], values=[1.0] * 6),
        ],
    )
    mn.check_model()
