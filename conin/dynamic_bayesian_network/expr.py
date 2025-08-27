import dataclasses


class ExpressionNode:

    def __add__(self, other):
        return AddNode(self, other)

    def __radd__(self, other):
        return AddNode(other, self)

    def __sub__(self, other):
        return MinusNode(self, other)

    def __rsub__(self, other):
        return MinusNode(other, self)


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

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
