"""
Pyomo walker for translating smoek expressions with ConinVarNode to Pyomo.

This module extends smoek's SmoekToPyomoExprWalker to handle ConinVarNode,
allowing conin variable references to be translated into actual Pyomo variables.
"""

from smoek.pymodel.pyomo.walkers import SmoekToPyomoExprWalker
from conin.smoek.bridge import ConinVarNode


class ConinPyomoWalker(SmoekToPyomoExprWalker):
    """
    Walker that translates smoek expressions with ConinVarNode to Pyomo expressions.

    Extends smoek's SmoekToPyomoExprWalker to handle the special case of ConinVarNode,
    which represents a reference to a conin model variable. All other expression nodes
    are handled by smoek's existing translation logic.

    Attributes:
        original_model: The conin model object with V() method for variable access
    """

    def __init__(self, original_model):
        """
        Initialize the walker with the original conin model.

        Args:
            original_model: Conin model object that has V(node, state) or
                           V(node, time, state) method for variable access
        """
        super().__init__()
        self.original_model = original_model

    def _visit(self, expr):
        """
        Visit an expression node and translate it to Pyomo.

        Handles ConinVarNode specially by calling original_model.V() to get
        the actual Pyomo variable. All other node types are delegated to
        smoek's walker.

        Args:
            expr: Expression node to visit

        Returns:
            Pyomo expression or variable
        """
        if isinstance(expr, ConinVarNode):
            # Translate ConinVarNode to actual Pyomo variable
            # via the original model's V() method
            if expr.time is None:
                # 2-arg form: model.V(node, state)
                pyomo_var = self.original_model.V(expr.node, expr.state)
            else:
                # 3-arg form: model.V(node, time, state)
                pyomo_var = self.original_model.V(expr.node, expr.time, expr.state)

            # Push result onto stack (following smoek's walker pattern)
            self._stack.append(pyomo_var)
        else:
            # Delegate all other node types to smoek's walker
            super()._visit(expr)


def translate_expression_to_pyomo(expr, pyomo_model, original_model):
    """
    Translate a smoek expression tree (with ConinVarNode leaves) to Pyomo.

    This is a convenience function that creates a walker and translates
    the expression in one call.

    Args:
        expr: Smoek expression tree (may contain ConinVarNode instances)
        pyomo_model: The Pyomo model being built
        original_model: The original conin model with V() method

    Returns:
        Pyomo expression suitable for adding as a constraint

    Example:
        >>> from conin.smoek.bridge import ConinVarNode
        >>> # Build expression: V("A", 0) + V("B", 1) <= 10
        >>> expr = ConinVarNode("A", 0) + ConinVarNode("B", 1) <= 10
        >>> pyomo_expr = translate_expression_to_pyomo(expr, pyomo_model, original_model)
    """
    walker = ConinPyomoWalker(original_model)
    return walker.walk(expr, decl=None, model=pyomo_model)
