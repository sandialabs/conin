Markov Networks
===============

This page describes how to create ``DiscreteMarkovNetwork`` instances and how
to add constraints using the same simple examples that appear in
``conin.markov_network.examples``.

Discrete Factors
----------------

A ``DiscreteFactor`` assigns non-negative weights to one or more nodes.
Factors can be specified either with a dictionary keyed by assignments or with a
flat list interpreted in the model's state order.

.. code-block:: python

   from conin.markov_network import DiscreteFactor

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

Creating a ``DiscreteMarkovNetwork``
------------------------------------

The smallest complete example in ``conin.markov_network.examples`` is
``example6_conin``. It builds a two-node Markov network with one unary factor
for each node and one pairwise factor.

.. code-block:: python

   from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor

   pgm = DiscreteMarkovNetwork()
   pgm.states = {"A": [0, 1], "B": [0, 1]}
   pgm.edges = [("A", "B")]

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

   pgm.factors = [f_a, f_b, f_ab]
   pgm.check_model()

For a slightly larger example, ``ABC_conin`` defines three variables with
pairwise interactions:

.. code-block:: python

   import numpy as np
   from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor

   pgm = DiscreteMarkovNetwork()
   pgm.states = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}
   pgm.edges = [("A", "B"), ("B", "C"), ("A", "C")]

   f_a = DiscreteFactor(nodes=["A"], values=[1, 1, 2])
   f_b = DiscreteFactor(nodes=["B"], values=[1, 1, 3])
   f_c = DiscreteFactor(nodes=["C"], values=[1, 2, 1])
   f_ab = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
   f_bc = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
   f_ac = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))

   pgm.factors = [f_a, f_b, f_c, f_ab, f_bc, f_ac]
   pgm.check_model()

Constrained Markov networks
---------------------------

``ConstrainedDiscreteMarkovNetwork`` wraps a base Markov network together with a
list of constraint functors. The examples in
``conin.markov_network.examples`` use the same three-variable ``ABC`` model and
add an all-different constraint.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``ABC_constrained_pyomo_conin`` uses ``@pyomo_constraint_fn`` to add algebraic
constraints to the optimization model:

.. code-block:: python

   from conin import pyomo_constraint_fn
   from conin.markov_network import ConstrainedDiscreteMarkovNetwork
   from conin.markov_network.examples import ABC_conin

   base = ABC_conin().pgm

   @pyomo_constraint_fn()
   def constraint_fn(model):
       @model.Constraint([0, 1, 2])
       def diff(m, s):
           return m.V("A", s) + m.V("B", s) + m.V("C", s) <= 1

   constrained = ConstrainedDiscreteMarkovNetwork(
       base,
       constraints=[constraint_fn],
   )

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``ABC_constrained_toulbar2_conin`` expresses the same all-different constraint
with the Toulbar2 interface:

.. code-block:: python

   from conin import toulbar2_constraint_fn
   from conin.markov_network import ConstrainedDiscreteMarkovNetwork
   from conin.markov_network.examples import ABC_conin

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
       constraints=[constraint_fn],
   )

Factor constraints
^^^^^^^^^^^^^^^^^^

``ABC_constrained_factor_conin`` creates the constraint as an auxiliary factor:

.. code-block:: python

   from conin import factor_constraint_fn
   from conin.markov_network import ConstrainedDiscreteMarkovNetwork
   from conin.markov_network.examples import ABC_conin

   base = ABC_conin().pgm

   @factor_constraint_fn(nodes=["A", "B", "C"])
   def constraint_fn(states):
       values = set(states.values())
       return len(values) == 3

   constrained = ConstrainedDiscreteMarkovNetwork(
       base,
       constraints=[constraint_fn],
   )

Notes
-----

- ``states`` defines the allowed values for each random variable.
- ``edges`` can be given explicitly or inferred from the factor scopes.
- ``check_model()`` is a good final step after assigning states and factors.
- Simpler examples such as ``example6_conin`` and ``ABC_conin`` are usually the
  best starting point for custom models.
