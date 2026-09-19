# Smoek Algebraic Modeling Extension for Conin

This package provides a smoek-inspired algebraic modeling interface for defining conin constraints using natural mathematical syntax.

## Overview

Instead of writing imperative constraint code:

```python
@pyomo_constraint_fn()
def old_style(model):
    model.c = pyo.ConstraintList()
    model.c.add(model.V("A", 0) + model.V("B", 0) <= 1)
```

You can now write algebraic expressions:

```python
@algebraic_pyomo_constraint_fn()
def new_style(model, data):
    return model.V("A", 0) + model.V("B", 0) <= 1
```

## Architecture

The extension implements **Alternative C (Hybrid Approach)** from the design plan:

1. **ConinVarNode** (`bridge.py`) - Extends `smoek.ExprLeaf` to bridge conin variables into smoek's expression system
2. **Walkers** (`walkers/`) - Translate smoek expressions to Pyomo/Toulbar2:
   - `ConinPyomoWalker` - Extends smoek's Pyomo walker
   - `ConinToulbar2Walker` - Custom walker for toulbar2 linear constraints
3. **Decorators** (`decorators.py`) - User-facing decorators:
   - `@algebraic_pyomo_constraint_fn()`
   - `@algebraic_toulbar2_constraint_fn()`

## Features

### Basic Algebraic Expressions

```python
from conin.smoek import algebraic_pyomo_constraint_fn

@algebraic_pyomo_constraint_fn()
def constraint(model, data):
    # Simple constraint
    return model.V("A", 0) + model.V("B", 1) <= 1

@algebraic_pyomo_constraint_fn()
def multi_constraints(model, data):
    # Multiple constraints
    return [
        model.V("A", s) + model.V("B", s) <= 1
        for s in [0, 1, 2]
    ]
```

### Arithmetic Operations

Supports all standard arithmetic operations:
- Addition: `model.V("A", 0) + model.V("B", 1)`
- Subtraction: `model.V("A", 0) - model.V("B", 1)`
- Multiplication: `2 * model.V("A", 0)`
- Division: `model.V("A", 0) / 2`

### Comparison Operators

- Less than or equal: `<=`
- Greater than or equal: `>=`
- Equality: `==`

### Complex Expressions

```python
@algebraic_pyomo_constraint_fn()
def complex_constraint(model, data):
    return 2 * model.V("A", 0) + 3 * model.V("B", 1) - model.V("C", 0) <= 10
```

### Using Smoek Features

The extension re-exports smoek components for advanced usage:

```python
from conin.smoek import algebraic_pyomo_constraint_fn, RangeSet, sum_

@algebraic_pyomo_constraint_fn()
def with_sets(model, data):
    states = RangeSet(0, 2)
    return [
        sum_(model.V(node, s) for s in states) <= 1
        for node in data.nodes
    ]
```

### Toulbar2 Support

```python
from conin.smoek import algebraic_toulbar2_constraint_fn

@algebraic_toulbar2_constraint_fn()
def toulbar_constraint(model, data):
    return model.V("A", 0) + model.V("B", 1) <= 1
```

### Mixing Constraint Styles

Old and new styles can coexist:

```python
cpgm = ConstrainedDiscreteMarkovNetwork(
    pgm,
    constraints=[
        old_style_constraint,     # Uses @pyomo_constraint_fn
        new_algebraic_constraint, # Uses @algebraic_pyomo_constraint_fn
    ]
)
```

## Implementation Details

### Expression Building Flow

1. Decorator wraps user function in `AlgebraicPyomoConstraint` or `AlgebraicToulbar2Constraint`
2. During inference, constraint's `__call__(model, data)` is invoked
3. Model is wrapped in `WrappedModel` proxy where `model.V()` returns `ConinVarNode`
4. User function executes, building smoek expression tree via operator overloading
5. Walker traverses tree and translates to target format (Pyomo/Toulbar2)
6. Constraints added to model

### How It Works

**ConinVarNode as Bridge:**
```python
class ConinVarNode(smoek.ExprLeaf):
    """Inherits all smoek operator overloading."""
    def __init__(self, node, state, time=None):
        self.node = node
        self.state = state
        self.time = time
```

**Walker Translation:**
```python
class ConinPyomoWalker(SmoekToPyomoExprWalker):
    def _visit(self, expr):
        if isinstance(expr, ConinVarNode):
            # Translate to actual Pyomo variable
            return self.original_model.V(expr.node, expr.state)
        else:
            # Delegate to smoek's walker
            return super()._visit(expr)
```

## Files

- `__init__.py` - Public API and smoek re-exports
- `bridge.py` - ConinVarNode bridge class (~60 lines)
- `decorators.py` - Decorator classes and WrappedModel (~230 lines)
- `walkers/__init__.py` - Walker package
- `walkers/pyomo.py` - Pyomo translation (~90 lines)
- `walkers/toulbar2.py` - Toulbar2 translation (~200 lines)
- `tests/` - Test suite

**Total: ~580 lines** of implementation code (vs ~900 to reimplement smoek features)

## Future Extensions

The architecture immediately supports (via smoek):

- **Sets**: `RangeSet`, `SequenceSet` for defining domains
- **Aggregates**: `sum_()`, `prod()` for aggregate expressions
- **Indexed variables**: Investigation needed for `x[i,j]` syntax
- **Forall/quantification**: Investigation needed for quantified constraints

## Benefits

1. **Natural syntax** - Write constraints like mathematical expressions
2. **Leverages smoek** - Gets mature expression system and future features
3. **Backward compatible** - Works alongside traditional constraints
4. **Minimal code** - Only ~580 lines to connect conin to smoek
5. **Well-separated** - Lives entirely in `conin/smoek/` subdirectory

## Testing

Run tests with pytest:
```bash
pytest conin/smoek/tests/
```

Or specific test files:
```bash
pytest conin/smoek/tests/test_integration.py -v
pytest conin/smoek/tests/test_basic.py -v
```

## Example: Converting Existing Constraints

**Before:**
```python
@pyomo_constraint_fn()
def cancer_constraints(model):
    model.c = pyo.ConstraintList()
    model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
    model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)
```

**After:**
```python
@algebraic_pyomo_constraint_fn()
def cancer_constraints(model, data):
    return [
        model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1,
        model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1,
    ]
```

Much more concise and readable!
