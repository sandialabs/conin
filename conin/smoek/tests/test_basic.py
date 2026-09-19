"""
Basic smoke test for the smoek algebraic modeling extension.
"""

import sys
import os

# Add project to path
sys.path.insert(0, '/projects/conin')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from conin.smoek import algebraic_pyomo_constraint_fn, algebraic_toulbar2_constraint_fn
    print("✓ Imported decorators")

    from conin.smoek import ConinVarNode
    print("✓ Imported ConinVarNode")

    from conin.smoek import RangeSet, SequenceSet, sum_
    print("✓ Imported smoek components")

    from conin.smoek.walkers.pyomo import ConinPyomoWalker
    print("✓ Imported Pyomo walker")

    from conin.smoek.walkers.toulbar2 import ConinToulbar2Walker
    print("✓ Imported Toulbar2 walker")

    print("\n✅ All imports successful!")


def test_coninvarnode():
    """Test ConinVarNode basic functionality."""
    print("\nTesting ConinVarNode...")

    from conin.smoek.bridge import ConinVarNode

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

    from conin.smoek.bridge import ConinVarNode

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

    from conin.smoek import algebraic_pyomo_constraint_fn

    @algebraic_pyomo_constraint_fn()
    def test_constraint(model, data):
        return model.V("A", 0) + model.V("B", 1) <= 1

    print(f"✓ Created decorated constraint: {test_constraint}")
    print(f"  Type: {type(test_constraint)}")
    print(f"  Name: {test_constraint.name}")

    from conin.smoek.decorators import AlgebraicPyomoConstraint
    assert isinstance(test_constraint, AlgebraicPyomoConstraint)
    print("✓ Constraint is correct type")

    print("\n✅ Decorator tests passed!")


def test_wrapped_model():
    """Test WrappedModel proxy."""
    print("\nTesting WrappedModel...")

    from conin.smoek.decorators import WrappedModel
    from conin.smoek.bridge import ConinVarNode

    # Create mock model
    class MockModel:
        def __init__(self):
            self.some_attr = "test"

    original = MockModel()
    wrapped = WrappedModel(original)

    # Test V() interception
    var = wrapped.V("A", 0)
    assert isinstance(var, ConinVarNode)
    assert var.node == "A"
    assert var.state == 0
    print("✓ V() returns ConinVarNode")

    # Test 3-arg form
    var3 = wrapped.V("X", 5, 1)
    assert isinstance(var3, ConinVarNode)
    assert var3.node == "X"
    assert var3.time == 5
    assert var3.state == 1
    print("✓ V() handles 3-arg form correctly")

    # Test attribute forwarding
    assert wrapped.some_attr == "test"
    print("✓ Attributes forwarded to original")

    print("\n✅ WrappedModel tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SMOEK ALGEBRAIC MODELING EXTENSION - BASIC TESTS")
    print("=" * 60)

    try:
        test_imports()
        test_coninvarnode()
        test_expression_building()
        test_decorator_creation()
        test_wrapped_model()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
