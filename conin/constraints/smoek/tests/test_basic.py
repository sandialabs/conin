"""
Basic smoke test for the smoek algebraic modeling extension.
"""

import pytest
from conin.util import try_import

with try_import() as smoek_available:
    import smoek  # noqa: F401

if not smoek_available:
    pytestmark = pytest.mark.skip(reason="Smoek not installed")


def test_coninvarnode():
    """Test ConinVarNode basic functionality."""
    from conin.constraints.smoek.bridge import ConinVarNode

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


def test_decorator_creation():
    """Test that decorators can be created."""
    from conin import smoek_constraint_fn

    @smoek_constraint_fn()
    def test_constraint(model, data):
        return model.V("A", 0) + model.V("B", 1) <= 1

    from conin.constraints.smoek.decorators import SmoekConstraint

    assert isinstance(test_constraint, SmoekConstraint)
