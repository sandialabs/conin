"""
Decorators for algebraic constraint definition using smoek expressions.

This module provides decorators that allow users to define constraints using
natural algebraic syntax (e.g., model.V("A", 0) + model.V("B", 1) <= 10)
which are then translated to Pyomo or Toulbar2 constraint declarations.
"""

import inspect
from conin.constraint import ConstraintFunctor
from conin.smoek.bridge import ConinVarNode
from conin.smoek.walkers.pyomo import translate_expression_to_pyomo
from conin.smoek.walkers.toulbar2 import translate_expression_to_toulbar2


class WrappedModel:
    """
    Proxy that wraps a model to intercept V() calls and return ConinVarNode.

    This allows user constraint functions to build smoek expression trees
    by calling model.V(...) which returns expression nodes instead of actual
    variables.
    """

    def __init__(self, original):
        """
        Initialize the wrapped model.

        Args:
            original: The original conin/pyomo/toulbar2 model to wrap
        """
        self._original = original

    def V(self, *args):
        """
        Intercept V() calls and return ConinVarNode instead of actual variable.

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

    def __getattr__(self, name):
        """Forward all other attribute access to the original model."""
        return getattr(self._original, name)


class AlgebraicPyomoConstraint(ConstraintFunctor):
    """
    Constraint functor for algebraic Pyomo constraints.

    Wraps a user function that returns smoek expression trees and translates
    them to Pyomo constraints.
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
            model: Pyomo model being constructed
            data: Optional data for constraint

        Returns:
            Modified model
        """
        # Wrap model so V() returns ConinVarNode
        wrapped_model = WrappedModel(model)

        # Call user function to get expression tree(s)
        if self.num_args == 1:
            result = self.func(wrapped_model)
        else:
            result = self.func(wrapped_model, data)

        # Handle single expression or list of expressions
        if isinstance(result, (list, tuple)):
            exprs = result
        else:
            exprs = [result]

        # Translate each expression to Pyomo and add to model
        try:
            import pyomo.environ as pyo
        except ImportError:
            raise ImportError("Pyomo is required for algebraic_pyomo_constraint_fn")

        # Create constraint list if not exists
        if not hasattr(model, self.name):
            setattr(model, self.name, pyo.ConstraintList())

        constraint_list = getattr(model, self.name)

        for expr in exprs:
            # Translate smoek expression to Pyomo
            pyomo_expr = translate_expression_to_pyomo(expr, model, model)
            # Add to constraint list
            constraint_list.add(pyomo_expr)

        return model


class AlgebraicToulbar2Constraint(ConstraintFunctor):
    """
    Constraint functor for algebraic Toulbar2 constraints.

    Wraps a user function that returns smoek expression trees and translates
    them to Toulbar2 linear constraints.
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
            model: Toulbar2 model being constructed
            data: Optional data for constraint

        Returns:
            Modified model
        """
        # Wrap model so V() returns ConinVarNode
        wrapped_model = WrappedModel(model)

        # Call user function to get expression tree(s)
        if self.num_args == 1:
            result = self.func(wrapped_model)
        else:
            result = self.func(wrapped_model, data)

        # Handle single expression or list of expressions
        if isinstance(result, (list, tuple)):
            exprs = result
        else:
            exprs = [result]

        # Translate each expression to Toulbar2 and add to model
        for expr in exprs:
            # Translate smoek expression to toulbar2 format
            var_terms, operator, rhs = translate_expression_to_toulbar2(expr, model)
            # Add linear constraint to model
            model.AddGeneralizedLinearConstraint(var_terms, operator, rhs)

        return model


def algebraic_pyomo_constraint_fn(*, name=None):
    """
    Decorator for defining algebraic Pyomo constraints.

    Allows users to define constraints using natural algebraic syntax with
    smoek expressions. The constraint function should return a smoek expression
    or list of expressions that will be translated to Pyomo constraints.

    Args:
        name: Optional name for the constraint (defaults to function name)

    Returns:
        Decorator function that wraps user function in AlgebraicPyomoConstraint

    Example:
        >>> from conin.smoek import algebraic_pyomo_constraint_fn
        >>>
        >>> @algebraic_pyomo_constraint_fn()
        >>> def my_constraint(model, data):
        >>>     # Natural algebraic syntax
        >>>     return model.V("A", 0) + model.V("B", 0) <= 1
        >>>
        >>> # Or multiple constraints
        >>> @algebraic_pyomo_constraint_fn()
        >>> def multi_constraint(model, data):
        >>>     return [
        >>>         model.V("A", s) + model.V("B", s) <= 1
        >>>         for s in [0, 1, 2]
        >>>     ]
    """
    def decorator(func):
        return AlgebraicPyomoConstraint(func=func, name=name)
    return decorator


def algebraic_toulbar2_constraint_fn(*, name=None):
    """
    Decorator for defining algebraic Toulbar2 constraints.

    Allows users to define constraints using natural algebraic syntax with
    smoek expressions. The constraint function should return a smoek expression
    or list of expressions that will be translated to Toulbar2 linear constraints.

    Args:
        name: Optional name for the constraint (defaults to function name)

    Returns:
        Decorator function that wraps user function in AlgebraicToulbar2Constraint

    Example:
        >>> from conin.smoek import algebraic_toulbar2_constraint_fn
        >>>
        >>> @algebraic_toulbar2_constraint_fn()
        >>> def my_constraint(model, data):
        >>>     # Natural algebraic syntax
        >>>     return model.V("A", 0) + model.V("B", 0) <= 1
        >>>
        >>> # With data
        >>> @algebraic_toulbar2_constraint_fn()
        >>> def budget_constraint(model, data):
        >>>     return sum(
        >>>         data.costs[s] * model.V(node, s)
        >>>         for node in data.nodes
        >>>         for s in [0, 1, 2]
        >>>     ) <= data.budget
    """
    def decorator(func):
        return AlgebraicToulbar2Constraint(func=func, name=name)
    return decorator
