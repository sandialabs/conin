"""
Smoek-like algebraic modeling extension for conin constraints.

This package provides decorators and utilities for defining conin constraints
using natural algebraic syntax powered by smoek's expression system.

Example:
    >>> from conin import smoek_constraint_fn
    >>>
    >>> @smoek_constraint_fn()
    >>> def my_constraint(model, data):
    >>>     return model.V("A", 0) + model.V("B", 0) <= 1

The package bridges conin's constraint system with smoek's algebraic modeling
capabilities, allowing users to write constraints with operator overloading
instead of imperative Pyomo/Toulbar2 code.
"""

from .decorators import smoek_constraint_fn, SmoekConstraint
from .add_constraints_oracle import create_oracle_constraints_from_smoek

__all__ = [
    "smoek_constraint_fn",
    "SmoekConstraint",
    "create_oracle_constraints_from_smoek",
]
