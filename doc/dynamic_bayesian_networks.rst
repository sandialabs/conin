Creating a Dynamic Discrete Bayesian Network
============================================

This page describes how to create a ``DynamicDiscreteBayesianNetwork`` instance using
``DiscreteCPD`` objects.
``DynamicDiscreteBayesianNetwork`` represents a two-slice dynamic Bayesian
network whose conditional probability distributions can reference the symbolic
time variable ``t``.

Dynamic Bayesian networks
-------------------------

The simplest example is ``simple0_DDBN_conin``, which contains one dynamic node
with an initial distribution and one transition CPD.

.. code-block:: python

   from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
   from conin.bayesian_network import DiscreteCPD

   dbn = DynamicDiscreteBayesianNetwork()
   dbn.dynamic_states = {"Z": [0, 1]}

   z_start = DiscreteCPD(node=("Z", 0), values=[0.5, 0.5])
   z_transition = DiscreteCPD(
       node=("Z", dbn.t),
       parents=[("Z", dbn.t - 1)],
       values={0: [0.7, 0.3], 1: [0.8, 0.2]},
   )

   dbn.cpds = [z_start, z_transition]
   dbn.check_model()

The ``simple1_DDBN_conin`` example introduces two dynamic nodes:

.. code-block:: python

   from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
   from conin.bayesian_network import DiscreteCPD

   dbn = DynamicDiscreteBayesianNetwork()
   dbn.dynamic_states = {"A": [0, 1], "B": [0, 1]}

   cpd_start_a = DiscreteCPD(node=("A", 0), values=[0.9, 0.1])
   cpd_start_b = DiscreteCPD(
       node=("B", dbn.t),
       parents=[("A", dbn.t)],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )
   cpd_trans_a = DiscreteCPD(
       node=("A", dbn.t),
       parents=[("A", dbn.t - 1)],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )

   dbn.cpds = [cpd_start_a, cpd_start_b, cpd_trans_a]
   dbn.check_model()

Constrained dynamic Bayesian networks
-------------------------------------

``ConstrainedDynamicDiscreteBayesianNetwork`` wraps a base dynamic Bayesian
network together with a list of constraints. The examples in
``conin.dynamic_bayesian_network.examples`` use the same decorators as the
other model families.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_pyomo_conin`` requires the values of ``A`` and ``B``
to stay fixed across the first two time steps:

.. code-block:: python

   import pyomo.environ as pyo
   from conin import pyomo_constraint_fn
   from conin.dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.dynamic_bayesian_network.examples import simple1_DDBN_conin

   base = simple1_DDBN_conin().pgm

   @pyomo_constraint_fn()
   def constraints(model):
       model.c = pyo.ConstraintList()
       model.c.add(model.V("A", 0, 0) == model.V("A", 1, 0))
       model.c.add(model.V("B", 0, 0) == model.V("B", 1, 0))

   constrained = ConstrainedDynamicDiscreteBayesianNetwork(
       base,
       constraints=[constraints],
   )

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_toulbar2_conin`` expresses the same condition with
Toulbar2:

.. code-block:: python

   from conin import toulbar2_constraint_fn
   from conin.dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.dynamic_bayesian_network.examples import simple1_DDBN_conin

   base = simple1_DDBN_conin().pgm

   @toulbar2_constraint_fn()
   def constraints(model):
       model.AddGeneralizedLinearConstraint(
           [model.V("A", 0, 0), model.V("A", 1, 0, coef=-1)],
           "==",
           0,
       )
       model.AddGeneralizedLinearConstraint(
           [model.V("B", 0, 0), model.V("B", 1, 0, coef=-1)],
           "==",
           0,
       )

   constrained = ConstrainedDynamicDiscreteBayesianNetwork(
       base,
       constraints=[constraints],
   )

Factor constraints
^^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_factor_conin`` creates a constraint over a generated
set of nodes:

.. code-block:: python

   from conin import factor_constraint_fn
   from conin.dynamic_bayesian_network import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.dynamic_bayesian_network.examples import simple1_DDBN_conin

   base = simple1_DDBN_conin().pgm

   def nodes(data):
       for t in range(data.T):
           yield ("A", t)
           yield ("B", t)

   @factor_constraint_fn(nodes=nodes)
   def constraints(states):
       return states["A", 0] == states["A", 1] and states["B", 0] == states["B", 1]

   constrained = ConstrainedDynamicDiscreteBayesianNetwork(
       base,
       constraints=[constraints],
   )

Notes
-----

- ``dynamic_states`` lists the state space for time-indexed variables.
- ``dbn.t`` is a symbolic time variable used when declaring repeated CPDs.
- Simpler models such as ``simple0_DDBN_conin`` and ``simple1_DDBN_conin`` are
  better documentation examples than the larger weather model.
