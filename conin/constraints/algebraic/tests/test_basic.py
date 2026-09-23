"""
Basic smoke test for the smoek algebraic modeling extension.
"""

import pytest
from conin.util import try_import

with try_import() as smoek_available:
    import smoek

if not smoek_available:
    pytestmark = pytest.mark.skip(reason="Smoek not installed")

import sys
import os

# Add project to path
sys.path.insert(0, "/projects/conin")


def test_imports():
    """Test that all modules can be imported."""
    from conin import algebraic_constraint_fn
    from conin import algebraic_constraint_fn as smoek_algebraic_constraint_fn

def test_coninvarnode():
    """Test ConinVarNode basic functionality."""
    from conin.constraints.algebraic.bridge import ConinVarNode

    # Test 2-arg form
    var1 = ConinVarNode("A", 0)
    assert var1.node == "A"
    assert var1.state == 0
    assert var1.time is None

    # Test 3-arg form (with time)
    var2 = ConinVarNode("X", 1, time=5)
    assert var2.node == "X"
    assert var2.state == 1
    assert var2.time == 5

    # Test operator overloading (inherited from smoek.ExprLeaf)
    expr = var1 + var2


def test_expression_building():
    """Test building simple algebraic expressions."""
    from conin.constraints.algebraic.bridge import ConinVarNode

    # Build expression: V("A", 0) + V("B", 1) <= 10
    a = ConinVarNode("A", 0)
    b = ConinVarNode("B", 1)

    # Test arithmetic
    sum_expr = a + b

    prod_expr = 2 * a

    # Test comparison
    constraint = a + b <= 10


def test_decorator_creation():
    """Test that decorators can be created."""
    from conin import algebraic_constraint_fn

    @algebraic_constraint_fn()
    def test_constraint(model, data):
        return model.V("A", 0) + model.V("B", 1) <= 1

    from conin.constraints.algebraic.decorators import AlgebraicConstraint

    assert isinstance(test_constraint, AlgebraicConstraint)
