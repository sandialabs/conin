"""
Example: Using the smoek algebraic modeling extension for conin constraints.

This example demonstrates the new algebraic syntax for defining constraints
compared to the traditional imperative syntax.
"""

import pyomo.environ as pyo
from conin.bayesian_network.examples import cancer1_BN_conin
from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
from conin.inference import map_query
from conin.constraint import pyomo_constraint_fn
from conin.smoek import algebraic_pyomo_constraint_fn, RangeSet, sum_


# ============================================================================
# Example 1: Simple Constraint
# ============================================================================

def example_simple_constraint():
    """Compare traditional vs algebraic syntax for a simple constraint."""
    print("=" * 70)
    print("Example 1: Simple Constraint")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    # Traditional syntax
    @pyomo_constraint_fn()
    def traditional(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)

    # Algebraic syntax (much cleaner!)
    @algebraic_pyomo_constraint_fn()
    def algebraic(model, data):
        return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1

    print("\nTraditional syntax (5 lines):")
    print("""
    @pyomo_constraint_fn()
    def traditional(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
    """)

    print("Algebraic syntax (3 lines):")
    print("""
    @algebraic_pyomo_constraint_fn()
    def algebraic(model, data):
        return model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1
    """)

    # Both produce same results
    evidence = {"Pollution": 0, "Smoker": 1}

    cpgm_trad = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[traditional])
    result_trad = map_query(cpgm_trad, method="integer_program", evidence=evidence)

    cpgm_alg = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[algebraic])
    result_alg = map_query(cpgm_alg, method="integer_program", evidence=evidence)

    print(f"\n✓ Traditional result: {result_trad}")
    print(f"✓ Algebraic result:   {result_alg}")
    print(f"✓ Results match: {result_trad == result_alg}")


# ============================================================================
# Example 2: Multiple Constraints
# ============================================================================

def example_multiple_constraints():
    """Show how algebraic syntax simplifies multiple constraints."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Constraints")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    # Traditional syntax
    @pyomo_constraint_fn()
    def traditional(model):
        model.c = pyo.ConstraintList()
        model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
        model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

    # Algebraic syntax - returns list
    @algebraic_pyomo_constraint_fn()
    def algebraic(model, data):
        return [
            model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
            model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
        ]

    print("\nAlgebraic syntax makes multiple constraints clearer:")
    print("""
    return [
        model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
        model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
    ]
    """)

    evidence = {"Pollution": 0, "Smoker": 1}

    cpgm_alg = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[algebraic])
    result = map_query(cpgm_alg, method="integer_program", evidence=evidence)

    print(f"✓ Result: {result}")


# ============================================================================
# Example 3: Complex Arithmetic
# ============================================================================

def example_complex_arithmetic():
    """Show complex arithmetic expressions."""
    print("\n" + "=" * 70)
    print("Example 3: Complex Arithmetic")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    @algebraic_pyomo_constraint_fn()
    def weighted_constraint(model, data):
        # Complex weighted sum with constants
        return (
            2 * model.V("Dyspnoea", 1) +
            3 * model.V("Xray", 1) -
            model.V("Cancer", 0)
            <= 3
        )

    print("\nComplex expression with weights:")
    print("""
    return (
        2 * model.V("Dyspnoea", 1) +
        3 * model.V("Xray", 1) -
        model.V("Cancer", 0)
        <= 3
    )
    """)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[weighted_constraint])
    evidence = {"Pollution": 0, "Smoker": 1}
    result = map_query(cpgm, method="integer_program", evidence=evidence)

    print(f"✓ Result: {result}")


# ============================================================================
# Example 4: List Comprehensions
# ============================================================================

def example_list_comprehension():
    """Show how algebraic syntax works well with list comprehensions."""
    print("\n" + "=" * 70)
    print("Example 4: List Comprehensions")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    @algebraic_pyomo_constraint_fn()
    def all_different(model, data):
        # Each state can be assigned to at most one variable
        return [
            model.V("Dyspnoea", s) + model.V("Xray", s) <= 1
            for s in [0, 1]
        ]

    print("\nList comprehension for pattern-based constraints:")
    print("""
    return [
        model.V("Dyspnoea", s) + model.V("Xray", s) <= 1
        for s in [0, 1]
    ]
    """)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[all_different])
    evidence = {"Pollution": 0, "Smoker": 1}
    result = map_query(cpgm, method="integer_program", evidence=evidence)

    print(f"✓ Result: {result}")


# ============================================================================
# Example 5: Using Smoek Features
# ============================================================================

def example_smoek_features():
    """Show how to use smoek's RangeSet and sum_ with conin constraints."""
    print("\n" + "=" * 70)
    print("Example 5: Using Smoek Features (RangeSet, sum_)")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    @algebraic_pyomo_constraint_fn()
    def with_smoek(model, data):
        # Use smoek's RangeSet for cleaner iteration
        states = RangeSet(0, 1)

        return [
            # Can use Python's sum()
            sum(model.V("Dyspnoea", s) for s in states) <= 1,

            # Or smoek's sum_ for more complex cases
            sum_(model.V("Xray", s) for s in states) <= 1,
        ]

    print("\nUsing smoek's RangeSet and sum_:")
    print("""
    from conin.smoek import algebraic_pyomo_constraint_fn, RangeSet, sum_

    @algebraic_pyomo_constraint_fn()
    def with_smoek(model, data):
        states = RangeSet(0, 1)

        return [
            sum(model.V("Dyspnoea", s) for s in states) <= 1,
            sum_(model.V("Xray", s) for s in states) <= 1,
        ]
    """)

    cpgm = ConstrainedDiscreteBayesianNetwork(pgm, constraints=[with_smoek])
    evidence = {"Pollution": 0, "Smoker": 1}
    result = map_query(cpgm, method="integer_program", evidence=evidence)

    print(f"✓ Result: {result}")


# ============================================================================
# Example 6: Mixing Old and New Styles
# ============================================================================

def example_mixing_styles():
    """Show that old and new constraint styles can coexist."""
    print("\n" + "=" * 70)
    print("Example 6: Mixing Traditional and Algebraic Styles")
    print("=" * 70)

    pgm = cancer1_BN_conin().pgm

    # Traditional style constraint
    @pyomo_constraint_fn()
    def traditional(model):
        model.c1 = pyo.ConstraintList()
        model.c1.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)

    # Algebraic style constraint
    @algebraic_pyomo_constraint_fn()
    def algebraic(model, data):
        return model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1

    # Mix both in same model!
    cpgm = ConstrainedDiscreteBayesianNetwork(
        pgm,
        constraints=[traditional, algebraic]
    )

    print("\nMixing constraints:")
    print("""
    cpgm = ConstrainedDiscreteBayesianNetwork(
        pgm,
        constraints=[
            traditional,  # Old style
            algebraic,    # New style
        ]
    )
    """)

    evidence = {"Pollution": 0, "Smoker": 1}
    result = map_query(cpgm, method="integer_program", evidence=evidence)

    print(f"✓ Result: {result}")
    print("✓ Both styles work together seamlessly!")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SMOEK ALGEBRAIC MODELING EXTENSION - EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the new algebraic syntax for")
    print("defining conin constraints using smoek's expression system.")
    print()

    try:
        example_simple_constraint()
        example_multiple_constraints()
        example_complex_arithmetic()
        example_list_comprehension()
        example_smoek_features()
        example_mixing_styles()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
