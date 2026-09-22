"""
Smoek-like algebraic modeling extension for conin constraints.

This package provides decorators and utilities for defining conin constraints
using natural algebraic syntax powered by smoek's expression system.

Example:
    >>> from conin import algebraic_pyomo_constraint_fn
    >>>
    >>> @algebraic_pyomo_constraint_fn()
    >>> def my_constraint(model, data):
    >>>     return model.V("A", 0) + model.V("B", 0) <= 1

The package bridges conin's constraint system with smoek's algebraic modeling
capabilities, allowing users to write constraints with operator overloading
instead of imperative Pyomo/Toulbar2 code.
"""

from .decorators import algebraic_constraint_fn, AlgebraicConstraint
from .bridge import ConinVarNode

# Re-export smoek components for user convenience
from smoek.core.model.set_components import RangeSet, SequenceSet, Set

# from smoek.core.expr.functions import sum as sum_, prod as prod

__all__ = [
    # Decorator
    "algebraic_constraint_fn",
    "AlgebraicConstriant",
    # Bridge
    "ConinVarNode",
    # Smoek components
    "RangeSet",
    "SequenceSet",
    "Set",
    #    'sum_',
    #    'prod',
]

__version__ = "0.1.0"
