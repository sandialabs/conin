"""Tests for conin.dynamic_bayesian_network.expr."""

from conin.dynamic_bayesian_network.expr import (
    AddNode,
    ExpressionConstant,
    ExpressionVariable,
    MinusNode,
)


class TestExpressionConstant:
    def test_value(self):
        c = ExpressionConstant(3)
        assert c.value() == 3

    def test_value_float(self):
        c = ExpressionConstant(1.5)
        assert c.value() == 1.5

    def test_value_zero(self):
        c = ExpressionConstant(0)
        assert c.value() == 0


class TestExpressionVariable:
    def test_initial_value_is_none(self):
        v = ExpressionVariable()
        assert v.value() is None

    def test_set_value(self):
        v = ExpressionVariable()
        v.set_value(7)
        assert v.value() == 7

    def test_set_value_overwrite(self):
        v = ExpressionVariable()
        v.set_value(1)
        v.set_value(2)
        assert v.value() == 2


class TestAddNode:
    def test_two_constants(self):
        node = AddNode(ExpressionConstant(2), ExpressionConstant(3))
        assert node.value() == 5

    def test_nested(self):
        # (1 + 2) + 3 == 6
        inner = AddNode(ExpressionConstant(1), ExpressionConstant(2))
        outer = AddNode(inner, ExpressionConstant(3))
        assert outer.value() == 6

    def test_with_variable(self):
        v = ExpressionVariable()
        v.set_value(10)
        node = AddNode(ExpressionConstant(5), v)
        assert node.value() == 15


class TestMinusNode:
    def test_two_constants(self):
        node = MinusNode(ExpressionConstant(5), ExpressionConstant(3))
        assert node.value() == 2

    def test_nested(self):
        # (10 - 3) - 2 == 5
        inner = MinusNode(ExpressionConstant(10), ExpressionConstant(3))
        outer = MinusNode(inner, ExpressionConstant(2))
        assert outer.value() == 5

    def test_with_variable(self):
        v = ExpressionVariable()
        v.set_value(4)
        node = MinusNode(ExpressionConstant(10), v)
        assert node.value() == 6


class TestExpressionNodeOperators:
    """Tests for the arithmetic operator overloads on ExpressionNode."""

    def test_add_two_nodes(self):
        a = ExpressionConstant(3)
        b = ExpressionConstant(4)
        result = a + b
        assert isinstance(result, AddNode)
        assert result.value() == 7

    def test_add_node_and_zero_returns_self(self):
        # Adding 0 (int) to an ExpressionNode should return the node itself.
        a = ExpressionConstant(5)
        result = a + 0
        assert result is a

    def test_radd_zero_returns_self(self):
        # sum() starts with 0 + first_element, exercising __radd__ with 0.
        a = ExpressionConstant(5)
        result = 0 + a
        assert result is a

    def test_add_node_and_scalar(self):
        a = ExpressionConstant(5)
        result = a + 3
        assert isinstance(result, AddNode)
        assert result.value() == 8

    def test_radd_scalar_and_node(self):
        a = ExpressionConstant(5)
        result = 3 + a
        assert isinstance(result, AddNode)
        assert result.value() == 8

    def test_sub_two_nodes(self):
        a = ExpressionConstant(7)
        b = ExpressionConstant(3)
        result = a - b
        assert isinstance(result, MinusNode)
        assert result.value() == 4

    def test_sub_node_and_zero_returns_self(self):
        a = ExpressionConstant(5)
        result = a - 0
        assert result is a

    def test_sub_node_and_scalar(self):
        a = ExpressionConstant(10)
        result = a - 4
        assert isinstance(result, MinusNode)
        assert result.value() == 6

    def test_rsub_scalar_and_node(self):
        a = ExpressionConstant(3)
        result = 10 - a
        assert isinstance(result, MinusNode)
        assert result.value() == 7

    def test_sum_builtin_uses_radd(self):
        # sum([a, b, c]) calls 0 + a then += b then += c; exercises __radd__ path.
        nodes = [ExpressionConstant(1), ExpressionConstant(2), ExpressionConstant(3)]
        result = sum(nodes)
        assert result.value() == 6
