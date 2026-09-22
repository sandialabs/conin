"""
Basic smoke test for the smoek algebraic modeling extension.
"""

import sys
import os

# Add project to path
sys.path.insert(0, "/projects/conin")


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from conin import algebraic_constraint_fn

    print("✓ Imported algebraic_constraint_fn from conin")

    from conin import algebraic_constraint_fn as smoek_algebraic_constraint_fn

    print("✓ Imported algebraic_constraint_fn from conin")

    from conin.constraints.smoek import ConinVarNode

    print("✓ Imported ConinVarNode")

    from conin.constraints.smoek import RangeSet, SequenceSet

    print("✓ Imported smoek components")

    print("\n✅ All imports successful!")


def test_coninvarnode():
    """Test ConinVarNode basic functionality."""
    print("\nTesting ConinVarNode...")

    from conin.constraints.smoek.bridge import ConinVarNode

    # Test 2-arg form
    var1 = ConinVarNode("A", 0)
    assert var1.node == "A"
    assert var1.state == 0
    assert var1.time is None
    print(f"✓ Created 2-arg ConinVarNode: {var1}")

    # Test 3-arg form (with time)
    var2 = ConinVarNode("X", 1, time=5)
    assert var2.node == "X"
    assert var2.state == 1
    assert var2.time == 5
    print(f"✓ Created 3-arg ConinVarNode: {var2}")

    # Test operator overloading (inherited from smoek.ExprLeaf)
    expr = var1 + var2
    print(f"✓ Operator overloading works: {var1} + {var2} = {type(expr)}")

    print("\n✅ ConinVarNode tests passed!")


def test_expression_building():
    """Test building simple algebraic expressions."""
    print("\nTesting expression building...")

    from conin.constraints.smoek.bridge import ConinVarNode

    # Build expression: V("A", 0) + V("B", 1) <= 10
    a = ConinVarNode("A", 0)
    b = ConinVarNode("B", 1)

    # Test arithmetic
    sum_expr = a + b
    print(f"✓ Addition: V(A,0) + V(B,1) = {type(sum_expr).__name__}")

    prod_expr = 2 * a
    print(f"✓ Multiplication: 2 * V(A,0) = {type(prod_expr).__name__}")

    # Test comparison
    constraint = a + b <= 10
    print(f"✓ Constraint: V(A,0) + V(B,1) <= 10 = {type(constraint).__name__}")

    print("\n✅ Expression building tests passed!")


def test_decorator_creation():
    """Test that decorators can be created."""
    print("\nTesting decorator creation...")

    from conin import algebraic_constraint_fn

    @algebraic_constraint_fn()
    def test_constraint(model, data):
        return model.V("A", 0) + model.V("B", 1) <= 1

    print(f"✓ Created decorated constraint: {test_constraint}")
    print(f"  Type: {type(test_constraint)}")
    print(f"  Name: {test_constraint.name}")

    from conin.constraints.smoek.decorators import AlgebraicConstraint

    assert isinstance(test_constraint, AlgebraicConstraint)
    print("✓ Constraint is correct type (AlgebraicConstraint)")

    print("\n✅ Decorator tests passed!")
