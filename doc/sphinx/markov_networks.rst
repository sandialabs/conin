Creating a Markov Network
=========================

This page describes how to create a ``DiscreteMarkovNetwork`` instance and how
to add constraints.

Discrete Factors
----------------

A ``DiscreteFactor`` assigns non-negative weights to one or more nodes.
Factors can be specified either with a dictionary keyed by assignments or with a
flat list interpreted in the model's state order.
For multi-node factors, dictionary values are the most explicit representation
and avoid ambiguity about Cartesian-product ordering.

.. testcode::

   from conin.graphical_models import DiscreteFactor

   f_a = DiscreteFactor(["A"], {0: 1, 1: 1})
   f_ab = DiscreteFactor(
       ["A", "B"],
       {
           (0, 0): 1,
           (0, 1): 3,
           (1, 0): 1,
           (1, 1): 1,
       },
   )

Markov networks
---------------

The smallest complete example in ``conin.examples.markov_network`` is
``example6_conin``. It builds a two-node Markov network with one unary factor
for each node and one pairwise factor.

.. testcode::

   from conin.graphical_models import DiscreteMarkovNetwork, DiscreteFactor

   f_a = DiscreteFactor(["A"], {0: 1, 1: 1})
   f_b = DiscreteFactor(["B"], {0: 1, 1: 2})
   f_ab = DiscreteFactor(
       ["A", "B"],
       {
           (0, 0): 1,
           (0, 1): 3,
           (1, 0): 1,
           (1, 1): 1,
       },
   )

   pgm = DiscreteMarkovNetwork(
       states={"A": [0, 1], "B": [0, 1]},
       edges=[("A", "B")],
       factors=[f_a, f_b, f_ab],
   )
   pgm.check_model()

For a slightly larger example, ``ABC_conin`` defines three variables with
pairwise interactions:

.. testcode::

   import numpy as np
   from conin.graphical_models import DiscreteMarkovNetwork, DiscreteFactor

   f_a = DiscreteFactor(nodes=["A"], values=[1, 1, 2])
   f_b = DiscreteFactor(nodes=["B"], values=[1, 1, 3])
   f_c = DiscreteFactor(nodes=["C"], values=[1, 2, 1])
   f_ab = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
   f_bc = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
   f_ac = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))

   pgm = DiscreteMarkovNetwork(
       states={"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]},
       edges=[("A", "B"), ("B", "C"), ("A", "C")],
       factors=[f_a, f_b, f_c, f_ab, f_bc, f_ac],
   )
   pgm.check_model()

Constrained Markov networks
---------------------------

``ConstrainedDiscreteMarkovNetwork`` wraps a base Markov network together with a
list of constraints. The examples in
``conin.examples.markov_network`` use the same three-variable ``ABC`` model and
add an all-different constraint.

The Pyomo and Toulbar2 examples require their respective solver backends. Oracle
constraints with ``nodes`` (``FactorOracleConstraint``) are represented as additional factors and can also be
consumed by variable-elimination workflows when ``pgmpy`` is installed.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``ABC_constrained_pyomo_conin`` uses ``@pyomo_constraint_fn`` to add algebraic
constraints to the optimization model:

.. testcode::

   from conin import pyomo_constraint_fn
   from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
   from conin.examples.markov_network import ABC_conin

   base = ABC_conin().pgm

   @pyomo_constraint_fn()
   def constraint_fn(model):
       @model.Constraint([0, 1, 2])
       def diff(m, s):
           return m.V("A", s) + m.V("B", s) + m.V("C", s) <= 1

   constrained = ConstrainedDiscreteMarkovNetwork(
       base,
       constraints=constraint_fn,
   )

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``ABC_constrained_toulbar2_conin`` expresses the same all-different constraint
with the Toulbar2 interface:

.. testcode::

   from conin import toulbar2_constraint_fn
   from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
   from conin.examples.markov_network import ABC_conin

   base = ABC_conin().pgm

   @toulbar2_constraint_fn()
   def constraint_fn(model):
       for value in [0, 1, 2]:
           model.AddGeneralizedLinearConstraint(
               [model.V("A", value), model.V("B", value), model.V("C", value)],
               "<=",
               1,
           )

   constrained = ConstrainedDiscreteMarkovNetwork(
       base,
       constraints=constraint_fn,
   )

Oracle constraints with node scoping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ABC_constrained_oracle_conin`` creates the constraint as an auxiliary factor:

.. testcode::

   from conin import oracle_constraint_fn
   from conin.constrained_models import ConstrainedDiscreteMarkovNetwork
   from conin.examples.markov_network import ABC_conin

   base = ABC_conin().pgm

   @oracle_constraint_fn(nodes=["A", "B", "C"])
   def constraint_fn(states):
       values = set(states.values())
       return len(values) == 3

   constrained = ConstrainedDiscreteMarkovNetwork(
       base,
       constraints=constraint_fn,
   )

Notes
-----

- ``factors`` is required and must be non-empty. Models are immutable after
  construction; use ``with_factor`` to get a new model with an extra factor.
- ``states`` defines the allowed values for each random variable.
- ``edges`` can be given explicitly or inferred from the factor scopes.
- ``check_model()`` is a good final step after constructing the model.
- See :doc:`model_conversion_io` for UAI file I/O and Bayesian-network to
  Markov-network conversion.
- Simpler examples such as ``example6_conin`` and ``ABC_conin`` are usually the
  best starting point for custom models.
