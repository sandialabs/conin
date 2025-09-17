Markov Network Model
====================

This page documents the discrete Markov network classes provided in
`conin/markov_network/model.py`. It explains core concepts, public attributes and
methods, and includes small, practical examples drawn from
`conin/markov_network/tests/examples.py`.


DiscreteFactor
--------------

Represents a factor over one or more discrete nodes.

- nodes: list of node identifiers (e.g., strings like ``"A"`` or integers).
- values: a dictionary or a list providing non-negative weights.
  - dict form: keys are assignments, e.g. ``{("A_val", "B_val"): weight}`` or ``{"A_val": weight}`` for unary.
  - list form: a flat list of weights that will be normalized against a model's state order via ``normalize``.
- default_value: optional fallback value for missing assignments (string or int, defaults to ``0``).

API Reference
^^^^^^^^^^^^^

.. autoclass:: conin.markov_network.DiscreteFactor
   :members:
   :undoc-members:
   :show-inheritance:

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


DiscreteMarkovNetwork
---------------------

A discrete Markov network defined by a set of nodes, optional edges, and a list of factors.

Construction
^^^^^^^^^^^^

- states: define node labels and their possible values, either as
  - list: node cardinalities (nodes become ``0..n-1``), or
  - dict: explicit mapping from node to list of values.
- edges: optional list of undirected edges ``[(u, v), ...]``. If omitted, edges are derived from factors.
- factors: list of ``DiscreteFactor`` instances. When set, they are normalized against the model's states.

API Reference
^^^^^^^^^^^^^

.. autoclass:: conin.markov_network.DiscreteMarkovNetwork
   :members:
   :undoc-members:
   :show-inheritance:

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


ConstrainedDiscreteMarkovNetwork
--------------------------------

A thin wrapper that augments a base ``DiscreteMarkovNetwork`` with user-defined optimization
constraints. The constraints are supplied as a functor that takes a Pyomo model and returns
the same model with constraints attached.

API Reference
^^^^^^^^^^^^^

.. autoclass:: conin.markov_network.ConstrainedDiscreteMarkovNetwork
   :members:
   :undoc-members:
   :show-inheritance:

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
