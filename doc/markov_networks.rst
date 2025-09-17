Markov Networks
===============

This page documents classes that are used to define discrete Markov networks.
The ``DiscreteFactor`` class is used to declare a discrete factor,
which is added to a ``DiscreteMarkovNetwork`` instance.  Further, the
``ConstrainedDiscreteMarkovNetwork`` class is used to augment a discrete
Markov network with application-specific constraints.


Discrete Factors
----------------

The ``DiscreteFactor`` class represents a factor defined over one or more discrete nodes. This class has the following public data members:

- nodes: a list of node identifiers (e.g., strings like ``"A"`` or integers).
- values: a dictionary or a list providing non-negative weights.
  - dict form: keys are assignments, e.g. ``{("A_val", "B_val"): weight}`` or ``{"A_val": weight}``.
  - list form: a list of weights that are associated with the Cartesian product of node states.
- default_value: optional fallback value for missing assignments (float, defaults to ``0``).

Example (unary and pairwise factors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conin.markov_network import DiscreteFactor

   # Unary factor on A with {0: 1, 1: 1}
   fA = DiscreteFactor(["A"], {0: 1, 1: 1})

   # Pairwise factor on (A, B)
   fAB = DiscreteFactor(["A", "B"], {
       (0, 0): 1,
       (0, 1): 3,
       (1, 0): 1,
       (1, 1): 1,
   })


Discrete Markov Networks
------------------------

A discrete Markov network defined by a set of nodes, optional edges, and a list of factors.

Construction
^^^^^^^^^^^^

- states: define node labels and their possible values, either as
  - list: node cardinalities (nodes become ``0..n-1``), or
  - dict: explicit mapping from node to list of values.
- edges: optional list of undirected edges ``[(u, v), ...]``. If omitted, edges are derived from factors.
- factors: list of ``DiscreteFactor`` instances. When set, they are normalized against the model's states.

Example (two-node network, adapted from ``example6_conin``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conin.markov_network import DiscreteMarkovNetwork, DiscreteFactor

   pgm = DiscreteMarkovNetwork()
   pgm.states = {"A": [0, 1], "B": [0, 1]}
   pgm.edges = [("A", "B")]

   fA = DiscreteFactor(["A"], {0: 1, 1: 1})
   fB = DiscreteFactor(["B"], {0: 1, 1: 2})
   fAB = DiscreteFactor(["A", "B"], {
       (0, 0): 1,
       (0, 1): 3,
       (1, 0): 1,
       (1, 1): 1,
   })

   pgm.factors = [fA, fB, fAB]

   # Optional: validate
   pgm.check_model()

   # Optional: build a MAP optimization model
   # model = pgm.create_map_query_model()
   # Solve with your preferred Pyomo solver (e.g., glpk, highs).


Constrained Discrete Markov Networks
------------------------------------

A thin wrapper that augments a base ``DiscreteMarkovNetwork`` with user-defined optimization
constraints. The constraints are supplied as a functor that takes a Pyomo model and returns
the same model with constraints attached.

Example (three nodes with pairwise interactions and a "values must differ" constraint)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from conin.markov_network import (
       DiscreteMarkovNetwork,
       DiscreteFactor,
       ConstrainedDiscreteMarkovNetwork,
   )

   # Base PGM (adapted from ABC_conin)
   base = DiscreteMarkovNetwork()
   base.states = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}
   base.edges = [("A", "B"), ("B", "C"), ("A", "C")]

   fA = DiscreteFactor(nodes=["A"], values=[1, 1, 2])
   fB = DiscreteFactor(nodes=["B"], values=[1, 1, 3])
   fC = DiscreteFactor(nodes=["C"], values=[1, 2, 1])
   fAB = DiscreteFactor(nodes=["A", "B"], values=np.ones(9))
   fBC = DiscreteFactor(nodes=["B", "C"], values=np.ones(9))
   fAC = DiscreteFactor(nodes=["A", "C"], values=np.ones(9))
   base.factors = [fA, fB, fC, fAB, fBC, fAC]

   # Constraint functor applied to the Pyomo model
   def constraint_fn(model):
       @model.Constraint([0, 1, 2])
       def all_different(m, s):
           # At most one variable can take the value s
           return m.X["A", s] + m.X["B", s] + m.X["C", s] <= 1
       return model

   constrained = ConstrainedDiscreteMarkovNetwork(base, constraints=constraint_fn)

   # Build the constrained MAP model
   # model = constrained.create_map_query_model()
   # Solve with your preferred Pyomo solver.


Notes and Behaviors
-------------------

- Factors must have non-negative weights; ``check_model`` asserts this.
- When factor ``values`` are provided as lists, they are normalized using the model's state order
  to a dictionary keyed by assignments.
- If ``edges`` are not set explicitly, they are inferred from factor scopes.
- ``create_map_query_model`` relies on Pyomo; ensure a compatible solver is installed to optimize the model.

