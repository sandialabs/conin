import dataclasses


class TimeExpression:
    """A time step written in terms of a :class:`TimeIndex`, such as ``t - 1``.

    ``+`` and ``-`` with integers or other time expressions build new
    expressions; ``value()`` evaluates them at the index's current value.
    """

    def __add__(self, other):
        if isinstance(other, TimeExpression):
            return AddNode(self, other)
        else:
            if other == 0:
                return self
            return AddNode(self, TimeConstant(other))

    def __radd__(self, other):
        if isinstance(other, TimeExpression):
            return AddNode(other, self)
        else:
            if other == 0:
                return self
            return AddNode(TimeConstant(other), self)

    def __sub__(self, other):
        if isinstance(other, TimeExpression):
            return MinusNode(self, other)
        else:
            if other == 0:
                return self
            return MinusNode(self, TimeConstant(other))

    def __rsub__(self, other):
        if isinstance(other, TimeExpression):
            return MinusNode(other, self)
        else:
            return MinusNode(TimeConstant(other), self)


@dataclasses.dataclass(frozen=True, slots=True)
class AddNode(TimeExpression):

    left: TimeExpression
    right: TimeExpression

    def value(self):
        return self.left.value() + self.right.value()


@dataclasses.dataclass(frozen=True, slots=True)
class MinusNode(TimeExpression):

    left: TimeExpression
    right: TimeExpression

    def value(self):
        return self.left.value() - self.right.value()


class TimeConstant(TimeExpression):

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class TimeIndex(TimeExpression):
    """The symbolic time index ``t`` of the CPD templates of a DBN.

    Create one, write the time-indexed CPDs against it (``("Z", t)``,
    ``("Z", t - 1)``) and pass it to
    :class:`~conin.graphical_models.DynamicDiscreteBayesianNetwork` as ``t``.
    Unrolling and sampling set its value with ``set_value`` while they
    evaluate the templates.
    """

    def __init__(self):
        self._value = None

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value
