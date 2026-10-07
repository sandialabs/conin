"""
Integration tests for the smoek algebraic modeling extension.

These tests verify end-to-end functionality by creating constrained models
and comparing results with the traditional constraint syntax.
"""

import pytest
from conin.util import try_import

with try_import() as smoek_available:
    import smoek

if not smoek_available:
    pytestmark = pytest.mark.skip(reason="Smoek not installed")


import pyomo.environ as pyo
from munch import Munch

from conin.bayesian_network.examples import cancer1_BN_conin
from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
from conin.inference import map_query
from conin import pyomo_constraint_fn, algebraic_constraint_fn
from conin.util import try_import

with try_import() as pytoulbar2_available:
    import pytoulbar2

skipif_toulbar2_not_available = pytest.mark.skipif(
    not pytoulbar2_available, reason="pytoulbar2 not installed"
)


class TestBasicAlgebraicConstraints:
    """Test basic algebraic constraint functionality."""

    def test_simple_pyomo_constraint(self):
        """Test that a simple algebraic constraint works with Pyomo."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            # Simple constraint: V("Dyspnoea", 1) + V("Xray", 1) <= 1
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )

        # Should be able to create inference model
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)

        # Verify we got a valid result
        assert result is not None
        assert "Cancer" in result.solution.states

    def test_multiple_pyomo_constraints(self):
        """Test algebraic constraint returning multiple constraints."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraints(model, data):
            # Return list of constraints
            return [
                model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
                model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
            ]

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraints]
        )

        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)

        assert result is not None
        assert "Cancer" in result.solution.states

    def test_comparison_with_traditional_syntax(self):
        """Test that algebraic and traditional syntax produce same results."""
        pgm = cancer1_BN_conin().pgm

        # Traditional syntax
        @pyomo_constraint_fn()
        def traditional_constraint(model):
            model.c = pyo.ConstraintList()
            model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
            model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

        # Algebraic syntax
        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            return [
                model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
                model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
            ]

        # Test both
        evidence = {"Pollution": 0, "Smoker": 1}

        cpgm_traditional = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[traditional_constraint]
        )
        result_traditional = map_query(
            cpgm_traditional, method="integer_program", evidence=evidence
        )

        cpgm_algebraic = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )
        result_algebraic = map_query(
            cpgm_algebraic, method="integer_program", evidence=evidence
        )

        # Results should be identical
        assert result_traditional.solution.states == result_algebraic.solution.states

    def test_mixing_constraint_styles(self):
        """Test that algebraic and traditional constraints can be mixed."""
        pgm = cancer1_BN_conin().pgm

        @pyomo_constraint_fn()
        def traditional_constraint(model):
            model.c1 = pyo.ConstraintList()
            model.c1.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            return model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1

        # Mix both styles
        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[traditional_constraint, algebraic_constraint]
        )

        with pytest.raises(AssertionError):
            evidence = {"Pollution": 0, "Smoker": 1}
            result = map_query(cpgm, method="integer_program", evidence=evidence)


class TestArithmeticOperations:
    """Test various arithmetic operations in algebraic constraints."""

    def test_addition(self):
        """Test addition operation."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None

    def test_subtraction(self):
        """Test subtraction operation."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) - model.V("Xray", 1) >= -1

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None

    def test_multiplication_by_constant(self):
        """Test multiplication by constant."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return 2 * model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 2

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None

    def test_complex_expression(self):
        """Test complex arithmetic expression."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return (
                2 * model.V("Dyspnoea", 1)
                + 3 * model.V("Xray", 1)
                - model.V("Cancer", 0)
                <= 3
            )

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None


class TestComparisonOperators:
    """Test different comparison operators."""

    def test_less_than_or_equal(self):
        """Test <= operator."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None

    def test_greater_than_or_equal(self):
        """Test >= operator."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) >= 0

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None

    def test_equality(self):
        """Test == operator."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) == 0

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None


class TestDataUsage:
    """Test using data parameter in constraints."""

    def test_constraint_with_data(self):
        """Test constraint that uses data parameter."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            # Use data if available
            limit = 1  # Default
            if hasattr(data, "limit"):
                limit = data.limit
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= limit

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="integer_program", evidence=evidence)
        assert result is not None


class TestToulbar2BasicAlgebraicConstraints:
    """Test basic algebraic constraint functionality with Toulbar2."""

    @skipif_toulbar2_not_available
    def test_simple_algebraic_constraint_toulbar2(self):
        """Test that a simple algebraic constraint works with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            # Simple constraint: V("Dyspnoea", 1) + V("Xray", 1) <= 1
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )

        # Should be able to create inference model
        evidence = {"Pollution": 0, "Smoker": 1}
        result = map_query(cpgm, method="toulbar2", evidence=evidence)

        # Verify we got a valid result
        assert result is not None
        assert "Cancer" in result.solution.states

    @skipif_toulbar2_not_available
    def test_comparison_pyomo_toulbar2(self):
        """Test that integer_program and toulbar2 produce same results for algebraic constraints."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )
        evidence = {"Pollution": 0, "Smoker": 1}

        # Solve with both methods
        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        # Results should be identical
        assert result_pyomo.solution.states == result_toulbar2.solution.states


class TestToulbar2ArithmeticOperations:
    """Test various arithmetic operations with both solvers."""

    @skipif_toulbar2_not_available
    def test_addition_toulbar2(self):
        """Test addition operation with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states

    @skipif_toulbar2_not_available
    def test_multiplication_by_constant_toulbar2(self):
        """Test multiplication by constant with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return 2 * model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 2

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states

    @skipif_toulbar2_not_available
    def test_complex_expression_toulbar2(self):
        """Test complex arithmetic expression with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return (
                2 * model.V("Dyspnoea", 1)
                + 3 * model.V("Xray", 1)
                - model.V("Cancer", 0)
                <= 3
            )

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states


class TestToulbar2ComparisonOperators:
    """Test different comparison operators with both solvers."""

    @skipif_toulbar2_not_available
    def test_less_than_or_equal_toulbar2(self):
        """Test <= operator with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states

    @skipif_toulbar2_not_available
    def test_greater_than_or_equal_toulbar2(self):
        """Test >= operator with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) + model.V("Xray", 1) >= 0

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states

    @skipif_toulbar2_not_available
    def test_equality_toulbar2(self):
        """Test == operator with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def constraint(model, data):
            return model.V("Dyspnoea", 1) == 0

        cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[constraint])
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states


class TestToulbar2MultipleConstraints:
    """Test multiple algebraic constraints with both solvers."""

    @skipif_toulbar2_not_available
    def test_multiple_constraints_toulbar2(self):
        """Test multiple algebraic constraints work with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            return [
                model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
                model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
            ]

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states

    @skipif_toulbar2_not_available
    def test_complex_multiple_constraints_toulbar2(self):
        """Test complex multiple algebraic constraints with Toulbar2."""
        pgm = cancer1_BN_conin().pgm

        @algebraic_constraint_fn()
        def algebraic_constraint(model, data):
            return [
                2 * model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 2,
                model.V("Dyspnoea", 0) + model.V("Xray", 0) >= 0,
            ]

        cpgm = ConstrainedDiscreteBayesianNetwork(
            pgm, constraints=[algebraic_constraint]
        )
        evidence = {"Pollution": 0, "Smoker": 1}

        result_pyomo = map_query(cpgm, method="integer_program", evidence=evidence)
        result_toulbar2 = map_query(cpgm, method="toulbar2", evidence=evidence)

        assert result_pyomo is not None
        assert result_toulbar2 is not None
        assert result_pyomo.solution.states == result_toulbar2.solution.states
