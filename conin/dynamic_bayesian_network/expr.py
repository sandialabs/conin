import dataclasses


class ExpressionNode:

    def __add__(self, other):
        if isinstance(other, ExpressionNode):
            return AddNode(self, other)
        else:
            if other == 0:
                return self
            return AddNode(self, ExpressionConstant(other))

    def __radd__(self, other):
        if isinstance(other, ExpressionNode):
            return AddNode(other, self)
        else:
            if other == 0:
                return self
            return AddNode(ExpressionConstant(other), self)

    def __sub__(self, other):
        if isinstance(other, ExpressionNode):
            return MinusNode(self, other)
        else:
            if other == 0:
                return self
            return MinusNode(self, ExpressionConstant(other))

    def __rsub__(self, other):
        if isinstance(other, ExpressionNode):
            return MinusNode(other, self)
        else:
            return MinusNode(ExpressionConstant(other), self)


@dataclasses.dataclass(frozen=True, slots=True)
class AddNode(ExpressionNode):

    left: ExpressionNode
    right: ExpressionNode

    def value(self):
        return self.left.value() + self.right.value()


@dataclasses.dataclass(frozen=True, slots=True)
class MinusNode(ExpressionNode):

    left: ExpressionNode
    right: ExpressionNode

    def value(self):
        return self.left.value() - self.right.value()


class ExpressionConstant(ExpressionNode):

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class ExpressionVariable(ExpressionNode):

    def __init__(self):
        self._value = None

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value
