"""Tests for ConstrainedDiscreteMarkovNetwork."""

import pytest

from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
from conin.constraints import pyomo_constraint_fn
from conin.exceptions import InvalidInputError

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


def _make_constraint():
    """A trivial Pyomo constraint (never applied in these tests)."""

    @pyomo_constraint_fn()
    def dummy(model):
        return model

    return dummy


# ---------------------------------------------------------------------------
# ConstrainedDiscreteMarkovNetwork tests
# ---------------------------------------------------------------------------


class TestConstrainedDiscreteMarkovNetwork:
    def test_wraps_model(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        assert cmn.model is mn

    def test_nodes_delegates_to_model(self):
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
        assert cmn.constraints == ()

    def test_constraints_kwarg(self):
        mn = _make_simple_mn()
        c = _make_constraint()
        cmn = ConstrainedDiscreteMarkovNetwork(mn, constraints=[c])
        assert isinstance(cmn.constraints, tuple)
        assert cmn.constraints == (c,)

    def test_constraints_kwarg_rejects_non_constraint(self):
        mn = _make_simple_mn()
        with pytest.raises(InvalidInputError):
            ConstrainedDiscreteMarkovNetwork(mn, constraints=[object()])

    def test_constraints_is_read_only(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        with pytest.raises(AttributeError):
            cmn.constraints = [_make_constraint()]
        assert cmn.constraints == ()

    def test_single_constraint_is_rejected(self):
        mn = _make_simple_mn()
        with pytest.raises(InvalidInputError):
            ConstrainedDiscreteMarkovNetwork(mn, constraints=_make_constraint())

    def test_string_constraints_rejected(self):
        mn = _make_simple_mn()
        with pytest.raises(InvalidInputError):
            ConstrainedDiscreteMarkovNetwork(mn, constraints="not-a-list")

    def test_check_model_delegates(self):
        mn = _make_simple_mn()
        cmn = ConstrainedDiscreteMarkovNetwork(mn)
        cmn.check_model()  # should not raise
