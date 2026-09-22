"""
Decorators for algebraic constraint definition using smoek expressions.

This module provides decorators that allow users to define constraints using
natural algebraic syntax (e.g., model.V("A", 0) + model.V("B", 1) <= 10)
which are then translated to Pyomo or Toulbar2 constraint declarations.
"""

import inspect
from conin.constraints import ConstraintFunctor
from conin.constraints.algebraic.bridge import ConinVarNode
from conin.util import try_import

with try_import() as smoek_available:
    import smoek


class ConinV:
    """
    A callable class that provides the V() interface for accessing conin variables.

    This class defines the V attribute that can be called to return
    ConinVarNode instances, allowing users to write constraint functions
    using smoek's algebraic syntax.
    """

    def __call__(self, *args):
        """
        Access conin variables as smoek expression nodes.

        Supports both 2-arg form V(node, state) and 3-arg form V(node, time, state).

        Args:
            *args: Either (node, state) or (node, time, state)

        Returns:
            ConinVarNode that can participate in smoek expression building

        Raises:
            ValueError: If number of arguments is not 2 or 3
        """
        if len(args) == 2:
            node, state = args
            return ConinVarNode(node, state)
        elif len(args) == 3:
            node, time, state = args
            return ConinVarNode(node, state, time=time)
        else:
            raise ValueError(f"V() requires 2 or 3 arguments, got {len(args)}")


class AlgebraicConstraint(ConstraintFunctor):
    """
    Unified constraint functor for algebraic constraints.

    Wraps a user function that adds constraints to a Smoek model.
    """

    def __init__(self, func, name=None):
        """
        Initialize the constraint.

        Args:
            func: User function that returns expression(s)
            name: Optional constraint name (defaults to function name)
        """
        self.func = func
        self.num_args = len(inspect.signature(self.func).parameters)
        if self.num_args > 2:
            raise ValueError("Algebraic constraint defined with more than 2 arguments")

        self.name = name if name else func.__name__

    def __call__(self, model, data):
        """
        Apply the constraint to the model.

        Args:
            model: Pyomo or Toulbar2 model being constructed
            data: Optional data for constraint

        Returns:
            Modified model
        """
        # Create a smoek model with V() method
        if hasattr(model, "V"):
            assert isinstance(
                model.V, ConinV
            ), f"A model attribute 'V' exists (model_type={type(model)} V_type={type(model.V)}). Smoek constraints reserve the 'V' attribute for access to Conin nodes"
        else:
            model.V = ConinV()

        # Call user function to get expression tree(s)
        if self.num_args == 1:
            result = self.func(model)
        else:
            result = self.func(model, data)

        if hasattr(model, "_conin_con_count"):
            count = model._conin_con_count
        else:
            count = model._conin_con_count = 0

        if (
            isinstance(result, smoek.core.model.constr_components.Constraint)
            or type(result) is smoek.core.expr.nodes.BinaryLogicalExprNode
        ):
            result = [result]

        if isinstance(result, list):
            for expr in result:
                count += 1
                if type(expr) is smoek.core.expr.nodes.BinaryLogicalExprNode:
                    con = smoek.constraint().expr(expr)
                else:
                    con = expr
                setattr(model, f"c_conin_{count}", con)

        model._conin_con_count = count


def algebraic_constraint_fn(*, name=None):
    """
    Decorator for defining algebraic constraints.

    Allows users to define constraints using natural algebraic syntax with
    smoek expressions. The constraint function should return a smoek expression
    or list of expressions. The inference backend automatically determines
    whether to translate to Pyomo or Toulbar2 format.

    Args:
        name: Optional name for the constraint (defaults to function name)

    Returns:
        Decorator function that wraps user function in AlgebraicConstraint

    Example:
        >>> from conin import algebraic_constraint_fn
        >>>
        >>> # Single constraints
        >>> @algebraic_constraint_fn()
        >>> def my_constraint(model, data):
        >>>     # Natural algebraic syntax - works with both Pyomo and Toulbar2
        >>>     return model.V("A", 0) + model.V("B", 0) <= 1
        >>>
        >>> # Multiple constraints in a list
        >>> @algebraic_constraint_fn()
        >>> def multi_constraint(model, data):
        >>>     return [
        >>>         model.V("A", s) + model.V("B", s) <= 1
        >>>         for s in [0, 1, 2]
        >>>     ]
        >>>
        >>> # Use with either inference method
        >>> cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[my_constraint])
        >>> result = map_query(cpgm, method="integer_program", evidence=...)  # Uses Pyomo
        >>> result = map_query(cpgm, method="toulbar2", evidence=...)  # Uses Toulbar2
    """

    def decorator(func):
        return AlgebraicConstraint(func=func, name=name)

    return decorator
