from conin.constraints.algebraic.add_constraints_toulbar2 import _scale_to_integers


def test_examples():
    """Test various coefficient scaling scenarios."""

    # Fractional binary powers — scale by 8
    coefficients = [0.5, 0.25, 0.125]
    rhs = 1.75
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [4, 2, 1]
    assert scaled_rhs == 14
    assert scale == 8.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)

    # Tenths — scale by 10
    coefficients = [0.1, 0.2, 0.3]
    rhs = 0.6
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [1, 2, 3]
    assert scaled_rhs == 6
    assert scale == 10.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)

    # Mixed decimals including three-decimal precision — scale by 1000
    coefficients = [1.5, 2.25, 0.333]
    rhs = 5.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [1500, 2250, 333]
    assert scaled_rhs == 5000
    assert scale == 1000.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)

    # Already integers — scale factor should be 1
    coefficients = [1.0, 2.0, 3.0]
    rhs = 6.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [1, 2, 3]
    assert scaled_rhs == 6
    assert scale == 1.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)

    # Milliths — scale by 1000, then GCD-reduce
    coefficients = [0.001, 0.002, 0.003]
    rhs = 0.01
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [1, 2, 3]
    assert scaled_rhs == 10
    assert scale == 1000.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)

    # Zero coefficient — should not affect GCD reduction
    coefficients = [1.5, 0.0, 2.5]
    rhs = 4.0
    scaled_coefs, scaled_rhs, scale = _scale_to_integers(coefficients, rhs)
    assert scaled_coefs == [3, 0, 5]
    assert scaled_rhs == 8
    assert scale == 2.0
    assert all(isinstance(c, int) for c in scaled_coefs)
    assert isinstance(scaled_rhs, int)
