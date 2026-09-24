from fractions import Fraction
from math import gcd
from functools import reduce
from conin.constraints.algebraic.add_constraints_toulbar2 import _scale_to_integers


def test_examples():
    """Test various coefficient scaling scenarios."""

    coefficients = [0.5, 0.25, 0.125]
    rhs = 1.75
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)

    coefficients = [0.1, 0.2, 0.3]
    rhs = 0.6
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)

    coefficients = [1.5, 2.25, 0.333]
    rhs = 5.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)

    coefficients = [1.0, 2.0, 3.0]
    rhs = 6.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)

    coefficients = [0.001, 0.002, 0.003]
    rhs = 0.01
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)

    coefficients = [1.5, 0.0, 2.5]
    rhs = 4.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
