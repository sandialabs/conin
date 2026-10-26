"""Tests for the toulbar2 constraint appliers.

``add_toulbar2_constraints`` turns the linear constraints of a Pyomo model into
``AddGeneralizedLinearConstraint`` calls; a recording stub stands in for the
pytoulbar2 CFN, so these tests do not need pytoulbar2.
"""

import pyomo.environ as pyo
import pytest
from pyomo.common.collections import ComponentMap

import conin.examples.markov_network as mex
from conin.constraints import (
    algebraic_constraint_fn,
    pyomo_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.representations.toulbar2.common import (
    Toulbar2VarWrapper,
    _scale_to_integers,
    add_toulbar2_constraints,
    apply_toulbar2_constraints,
)
from conin.util import try_import

with try_import() as smoek_available:
    import smoek  # noqa: F401


class _RecordingCFN:
    def __init__(self, V=None):
        self.V = V
        self.calls = []

    def AddGeneralizedLinearConstraint(self, variables, operator, rhs):
        self.calls.append((variables, operator, rhs))


def _pyomo_model(*constraints):
    """A Pyomo model over x[0..2]; x[i] stands for the toulbar2 (var i, value 1)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1, 2], within=pyo.Binary)
    m.V_to_tb2 = ComponentMap((m.x[i], (i, 1, 1)) for i in range(3))
    for k, expr in enumerate(constraints):
        setattr(m, f"c{k}", pyo.Constraint(expr=expr(m)))
    return m


def _calls(*constraints):
    cfn = _RecordingCFN()
    assert add_toulbar2_constraints(cfn, _pyomo_model(*constraints)) is cfn
    return cfn.calls


def test_equality():
    assert _calls(lambda m: m.x[0] + 2 * m.x[1] == 1) == [
        ([(0, 1, 1), (1, 1, 2)], "==", 1)
    ]


def test_upper_bound():
    assert _calls(lambda m: m.x[0] + m.x[2] <= 1) == [([(0, 1, 1), (2, 1, 1)], "<=", 1)]


def test_lower_bound():
    assert _calls(lambda m: m.x[0] + m.x[1] + m.x[2] >= 2) == [
        ([(0, 1, 1), (1, 1, 1), (2, 1, 1)], ">=", 2)
    ]


def test_range_is_split_in_two():
    assert _calls(lambda m: pyo.inequality(1, m.x[0] + m.x[1], 2)) == [
        ([(0, 1, 1), (1, 1, 1)], ">=", 1),
        ([(0, 1, 1), (1, 1, 1)], "<=", 2),
    ]


def test_constant_moves_to_the_right_hand_side():
    assert _calls(lambda m: m.x[0] + 3 <= 4) == [([(0, 1, 1)], "<=", 1)]


def test_fractional_coefficients_are_scaled_to_integers():
    assert _calls(lambda m: 0.5 * m.x[0] + 0.25 * m.x[1] <= 0.5) == [
        ([(0, 1, 2), (1, 1, 1)], "<=", 2)
    ]


def test_inactive_constraints_are_skipped():
    m = _pyomo_model(lambda m: m.x[0] <= 1, lambda m: m.x[1] >= 1)
    m.c0.deactivate()
    cfn = _RecordingCFN()
    add_toulbar2_constraints(cfn, m)
    assert cfn.calls == [([(1, 1, 1)], ">=", 1)]


def test_indexed_constraints():
    m = _pyomo_model()
    m.c = pyo.Constraint([0, 1], rule=lambda m, i: m.x[i] + m.x[2] <= 1)
    cfn = _RecordingCFN()
    add_toulbar2_constraints(cfn, m)
    assert cfn.calls == [
        ([(0, 1, 1), (2, 1, 1)], "<=", 1),
        ([(1, 1, 1), (2, 1, 1)], "<=", 1),
    ]


@pytest.mark.parametrize(
    "expr,match",
    [
        (lambda m: m.x[0] * m.x[1] <= 1, "Quadratic"),
        (lambda m: pyo.exp(m.x[0]) <= 1, "Nonlinear"),
    ],
)
def test_nonlinear_constraints_raise(expr, match):
    with pytest.raises(ValueError, match=match):
        _calls(expr)


def test_a_cfn_without_linear_constraints_raises_not_implemented():
    class OldCFN:
        pass

    with pytest.raises(NotImplementedError, match="AddGeneralizedLinearConstraint"):
        add_toulbar2_constraints(OldCFN(), _pyomo_model(lambda m: m.x[0] <= 1))


def test_cfn_errors_are_reraised_with_the_constraint_name():
    class FailingCFN:
        def AddGeneralizedLinearConstraint(self, *args):
            raise ValueError("boom")

    with pytest.raises(RuntimeError, match=r"'c0\[None\]'.*boom"):
        add_toulbar2_constraints(FailingCFN(), _pyomo_model(lambda m: m.x[0] <= 1))


@pytest.mark.parametrize(
    "coefficients,rhs,expected",
    [
        ([1.0, 2.0], 3.0, ([1, 2], 3, 1.0)),
        ([2.0, 4.0], 6.0, ([1, 2], 3, 0.5)),
        ([0.5, 0.25], 0.5, ([2, 1], 2, 4.0)),
        ([0.0, 1.0], 0.0, ([0, 1], 0, 1.0)),
    ],
)
def test_scale_to_integers(coefficients, rhs, expected):
    assert _scale_to_integers(coefficients, rhs) == expected


# ---------------------------------------------------------------------
# apply_toulbar2_constraints
# ---------------------------------------------------------------------


def test_apply_without_constraints_returns_the_model():
    cfn = _RecordingCFN()
    assert (
        apply_toulbar2_constraints(pgm=None, constraints=(), model=cfn, data=None)
        is cfn
    )


def test_apply_calls_the_toulbar2_constraints_in_order():
    seen = []

    @toulbar2_constraint_fn()
    def first(M, D):
        seen.append(("first", D))
        return M

    @toulbar2_constraint_fn()
    def second(M, D):
        seen.append(("second", D))
        return M

    cfn = _RecordingCFN()
    apply_toulbar2_constraints(pgm=None, constraints=(first, second), model=cfn, data=7)
    assert seen == [("first", 7), ("second", 7)]


def test_apply_rejects_mixed_constraint_lists():
    @toulbar2_constraint_fn()
    def tb2(M, D):
        return M

    @pyomo_constraint_fn()
    def pyomo(M, D):
        return M

    with pytest.raises(TypeError, match="all subsequent constraints"):
        apply_toulbar2_constraints(
            pgm=None, constraints=(tb2, pyomo), model=_RecordingCFN(), data=None
        )


def test_apply_rejects_other_constraint_types():
    @pyomo_constraint_fn()
    def pyomo(M, D):
        return M

    with pytest.raises(TypeError, match="Only Toulbar2Constraint"):
        apply_toulbar2_constraints(
            pgm=None, constraints=(pyomo,), model=_RecordingCFN(), data=None
        )


@pytest.mark.skipif(not smoek_available, reason="smoek not installed")
def test_apply_algebraic_constraints_gives_linear_constraints():
    # A and B are not both 0 (ABC_conin: states 0..2).
    pgm = mex.ABC_conin().pgm

    @algebraic_constraint_fn()
    def at_most_one_zero(model):
        return model.V("A", 0) + model.V("B", 0) <= 1

    cfn = _RecordingCFN(V=Toulbar2VarWrapper(pgm))
    apply_toulbar2_constraints(
        pgm=pgm, constraints=(at_most_one_zero,), model=cfn, data=None
    )
    a, b = pgm.nodes.index("A"), pgm.nodes.index("B")
    assert cfn.calls == [([(a, 0, 1), (b, 0, 1)], "<=", 1)]
