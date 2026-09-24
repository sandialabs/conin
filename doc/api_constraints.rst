Constraint Declarations
=======================

The ``conin.constraints`` package contains the constraint functors used to
create constrained graphical models. Most user code should create these objects
through the decorators below instead of instantiating the classes directly.

Constraint classes
------------------

.. autoclass:: conin.constraints.ConstraintFunctor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraints.OracleConstraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraints.PyomoConstraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraints.Toulbar2Constraint
   :members:
   :undoc-members:
   :show-inheritance:

Constraint decorators
---------------------

.. autofunction:: conin.constraints.oracle_constraint_fn

.. autofunction:: conin.constraints.pyomo_constraint_fn

.. autofunction:: conin.constraints.toulbar2_constraint_fn

Algebraic Constraints With Smoek
--------------------------------

``algebraic_constraint_fn`` is the high-level constraint interface for writing
linear constraints once and using them with either the Pyomo or Toulbar2
inference backend. It uses ``smoek`` expression objects under the hood so that
``model.V(...)`` references can participate in ordinary Python algebraic
expressions.

Use the two-argument ``model.V(node, state)`` form for static graphical models:

.. code-block:: python

   from conin import algebraic_constraint_fn

   @algebraic_constraint_fn()
   def at_most_one_active(model, data):
       return model.V("A", 1) + model.V("B", 1) <= 1

Use the three-argument ``model.V(node, time, state)`` form for dynamic models
and hidden Markov models:

.. code-block:: python

   from conin import algebraic_constraint_fn

   @algebraic_constraint_fn()
   def between_ten_and_twelve_h0(model, data):
       num_h0 = sum(model.V("H", t, "h0") for t in data.hmm.T)
       return [
           num_h0 >= 10,
           num_h0 <= 12,
       ]

The decorated function may return one expression or a list of expressions. The
inference backend translates those expressions into the backend-specific form at
solve time:

- ``map_query(..., method="integer_program")`` translates algebraic constraints
  to Pyomo constraints.
- ``map_query(..., method="toulbar2")`` translates supported linear algebraic
  constraints to Toulbar2 constraints.

Algebraic constraints require ``smoek`` to be installed. They are intended for
linear constraints over CONIN indicator variables; nonlinear or unsupported
expressions may fail during backend translation.

.. autoclass:: conin.constraints.AlgebraicConstraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: conin.constraints.algebraic_constraint_fn

Specialized Modules
-------------------

Reusable oracle constraints live in ``conin.constraints.oracle``. The
Toulbar2 helper module also includes utilities for translating supported linear
algebraic constraints to Toulbar2 expressions.

.. automodule:: conin.constraints.algebraic.add_constraints_toulbar2
   :members:
   :undoc-members:
