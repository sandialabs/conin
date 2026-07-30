Constraint Declarations
=======================

The following classes define constraint functors that used to create constraints in graphical models.
These classes are created using the constraint decorators below.

Constraint classes
------------------

.. autoclass:: conin.constraint.ConstraintFunctor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraint.OracleConstraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraint.PyomoConstraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraint.Toulbar2Constraint
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: conin.constraint.FactorConstraint
   :members:
   :undoc-members:
   :show-inheritance:

Constraint decorators
---------------------

.. autofunction:: conin.constraint.oracle_constraint_fn

.. autofunction:: conin.constraint.pyomo_constraint_fn

.. autofunction:: conin.constraint.toulbar2_constraint_fn

.. autofunction:: conin.constraint.factor_constraint_fn
