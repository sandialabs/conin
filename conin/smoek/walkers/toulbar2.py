"""
Toulbar2 walker for translating smoek expressions with ConinVarNode to toulbar2.

This module provides a walker that translates smoek expressions into toulbar2's
linear constraint format: AddGeneralizedLinearConstraint([vars], op, rhs).
"""

import smoek.core.expr.nodes
from smoek.core.expr.nodes import ExpressionType
from smoek.core.utils import BottomUpDepthFirstExpressionWalker
from conin.smoek.bridge import ConinVarNode


class ConinToulbar2Walker(BottomUpDepthFirstExpressionWalker):
    """
    Walker that translates smoek expressions with ConinVarNode to toulbar2 format.

    Toulbar2 requires linear constraints in the form:
        sum(coef_i * var_i) op rhs
    where op is '<=', '>=', or '=='

    This walker flattens smoek expression trees into this linear form.
    """

    def __init__(self, original_model):
        """
        Initialize the walker with the original conin model.

        Args:
            original_model: Conin model object that has V(node, state) method
                           which returns (var_id, state_id, coef) tuples
        """
        super().__init__()
        self.original_model = original_model
        self._stack = []

    def walk(self, expr):
        """
        Walk the expression tree and translate to toulbar2 format.

        Args:
            expr: Smoek expression tree (should be a constraint with <=, >=, ==)

        Returns:
            Tuple of (var_terms, operator, rhs) suitable for
            AddGeneralizedLinearConstraint(var_terms, operator, rhs)

        Raises:
            ValueError: If expression is not a linear constraint
        """
        assert self._stack == []
        self._walk(expr)
        result = self._stack.pop(0)
        assert self._stack == []
        return result

    def _visit(self, expr):
        """
        Visit an expression node and process it.

        Builds up linear constraint representation from expression tree.
        """
        if isinstance(expr, smoek.core.expr.nodes.BinaryExprNode):
            if expr.operation in [ExpressionType.eq, ExpressionType.leq, ExpressionType.geq]:
                # This is a constraint node
                right = self._stack.pop()
                left = self._stack.pop()

                # Determine operator
                if expr.operation == ExpressionType.eq:
                    op = "=="
                elif expr.operation == ExpressionType.leq:
                    op = "<="
                elif expr.operation == ExpressionType.geq:
                    op = ">="

                # Extract linear form: move everything to left side
                # left op right => left - right op 0
                var_terms, lhs_const = self._linearize(left)
                right_terms, rhs_const = self._linearize(right)

                # Combine: left - right op 0
                for var_id, state_id, coef in right_terms:
                    var_terms.append((var_id, state_id, -coef))

                rhs = rhs_const - lhs_const

                # Filter out zero coefficients and build V() result list
                var_list = []
                for var_id, state_id, coef in var_terms:
                    if coef != 0:
                        # Return in format model.V expects
                        var_list.append((var_id, state_id, coef))

                self._stack.append((var_list, op, rhs))

            else:
                # Arithmetic operation
                right = self._stack.pop()
                left = self._stack.pop()

                if expr.operation == ExpressionType.add:
                    result = self._add(left, right)
                elif expr.operation == ExpressionType.sub:
                    result = self._sub(left, right)
                elif expr.operation == ExpressionType.mul:
                    result = self._mul(left, right)
                elif expr.operation == ExpressionType.div:
                    result = self._div(left, right)
                else:
                    raise NotImplementedError(f"Operation {expr.operation} not supported for toulbar2")

                self._stack.append(result)

        elif isinstance(expr, ConinVarNode):
            # Variable reference - get (var_id, state_id, coef) from model
            if expr.time is None:
                var_tuple = self.original_model.V(expr.node, expr.state)
            else:
                var_tuple = self.original_model.V(expr.node, expr.time, expr.state)

            # Store as linear term: (var_terms, constant)
            self._stack.append(([var_tuple], 0))

        elif isinstance(expr, smoek.core.expr.nodes.NumberWrapper):
            # Constant - no variables, just the value
            self._stack.append(([], expr.value))

        else:
            raise NotImplementedError(
                f"Expression node {expr} of type {type(expr)} not supported for toulbar2"
            )

    def _linearize(self, term):
        """
        Extract linear terms and constant from a term.

        Args:
            term: Tuple of (var_terms, constant)

        Returns:
            Tuple of (var_terms, constant)
        """
        return term

    def _add(self, left, right):
        """Add two linear expressions."""
        left_terms, left_const = left
        right_terms, right_const = right
        return (left_terms + right_terms, left_const + right_const)

    def _sub(self, left, right):
        """Subtract two linear expressions."""
        left_terms, left_const = left
        right_terms, right_const = right
        # Negate right side coefficients
        negated_right = [(v, s, -c) for v, s, c in right_terms]
        return (left_terms + negated_right, left_const - right_const)

    def _mul(self, left, right):
        """Multiply linear expression by constant (only supports scalar multiplication)."""
        left_terms, left_const = left
        right_terms, right_const = right

        # Only support multiplication by constants
        if len(left_terms) == 0:
            # left is constant, multiply right
            return ([(v, s, left_const * c) for v, s, c in right_terms], left_const * right_const)
        elif len(right_terms) == 0:
            # right is constant, multiply left
            return ([(v, s, right_const * c) for v, s, c in left_terms], left_const * right_const)
        else:
            raise ValueError("Toulbar2 only supports linear constraints - cannot multiply two variable expressions")

    def _div(self, left, right):
        """Divide linear expression by constant."""
        left_terms, left_const = left
        right_terms, right_const = right

        # Only support division by constants
        if len(right_terms) == 0 and right_const != 0:
            return ([(v, s, c / right_const) for v, s, c in left_terms], left_const / right_const)
        else:
            raise ValueError("Toulbar2 only supports division by non-zero constants")


def translate_expression_to_toulbar2(expr, model):
    """
    Translate a smoek expression tree to toulbar2 constraint format.

    Args:
        expr: Smoek expression tree (should be a constraint)
        model: Toulbar2 model with V() method

    Returns:
        Tuple of (var_terms, operator, rhs) suitable for
        AddGeneralizedLinearConstraint()

    Example:
        >>> from conin.smoek.bridge import ConinVarNode
        >>> # Build expression: V("A", 0) + V("B", 1) <= 10
        >>> expr = ConinVarNode("A", 0) + ConinVarNode("B", 1) <= 10
        >>> var_terms, op, rhs = translate_expression_to_toulbar2(expr, model)
        >>> model.AddGeneralizedLinearConstraint(var_terms, op, rhs)
    """
    walker = ConinToulbar2Walker(model)
    return walker.walk(expr)
