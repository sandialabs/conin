"""
Smoek-like algebraic modeling extension for conin constraints.

This package provides decorators and utilities for defining conin constraints
using natural algebraic syntax powered by smoek's expression system.

Example:
    >>> from conin import algebraic_constraint_fn
    >>>
    >>> @algebraic_constraint_fn()
    >>> def my_constraint(model, data):
    >>>     return model.V("A", 0) + model.V("B", 0) <= 1

The package bridges conin's constraint system with smoek's algebraic modeling
capabilities, allowing users to write constraints with operator overloading
instead of imperative Pyomo/Toulbar2 code.
"""

from .decorators import algebraic_constraint_fn, AlgebraicConstraint

__all__ = [
    "algebraic_constraint_fn",
    "AlgebraicConstriant",
]
