"""Tests for ConstrainedDiscreteMarkovNetwork."""

import pytest

from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork

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
