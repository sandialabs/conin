Creating a Dynamic Discrete Bayesian Network
============================================

This page describes how to create a ``DynamicDiscreteBayesianNetwork`` instance using
``DiscreteCPD`` objects.
``DynamicDiscreteBayesianNetwork`` represents a two-slice dynamic Bayesian
network whose conditional probability distributions can reference the symbolic
time variable ``t``. Create an ``ExpressionVariable`` for ``t``, use it to
define the CPDs, and pass the same object to the constructor, which also
requires a non-empty list of CPDs and at least one of ``states`` or
``dynamic_states``.

Dynamic Bayesian networks
-------------------------

The simplest example is ``simple0_DDBN_conin``, which contains one dynamic node
with an initial distribution and one transition CPD.

.. testcode::

   from conin.graphical_models import (
       DynamicDiscreteBayesianNetwork,
       ExpressionVariable,
   )
   from conin.graphical_models import DiscreteCPD

   t = ExpressionVariable()

   z_start = DiscreteCPD(node=("Z", 0), values=[0.5, 0.5])
   z_transition = DiscreteCPD(
       node=("Z", t),
       parents=[("Z", t - 1)],
       values={0: [0.7, 0.3], 1: [0.8, 0.2]},
   )

   dbn = DynamicDiscreteBayesianNetwork(
       dynamic_states={"Z": [0, 1]}, cpds=[z_start, z_transition], t=t
   )
   dbn.check_model()

``check_model()`` is currently a lightweight consistency hook for dynamic
Bayesian networks. Static Bayesian networks produced from a dynamic network are
validated when they are unrolled for inference.

The ``simple1_DDBN_conin`` example introduces two dynamic nodes:

.. testcode::

   from conin.graphical_models import (
       DynamicDiscreteBayesianNetwork,
       ExpressionVariable,
   )
   from conin.graphical_models import DiscreteCPD

   t = ExpressionVariable()

   cpd_start_a = DiscreteCPD(node=("A", 0), values=[0.9, 0.1])
   cpd_start_b = DiscreteCPD(
       node=("B", t),
       parents=[("A", t)],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )
   cpd_trans_a = DiscreteCPD(
       node=("A", t),
       parents=[("A", t - 1)],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )

   dbn = DynamicDiscreteBayesianNetwork(
       dynamic_states={"A": [0, 1], "B": [0, 1]},
       cpds=[cpd_start_a, cpd_start_b, cpd_trans_a],
       t=t,
   )
   dbn.check_model()

Constrained dynamic Bayesian networks
-------------------------------------

``ConstrainedDynamicDiscreteBayesianNetwork`` wraps a base dynamic Bayesian
network together with a list of constraints. The examples in
``conin.examples.dynamic_bayesian_network`` use the same decorators as the
other model families.

The Pyomo and Toulbar2 examples require their respective solver backends. Oracle
constraints with ``nodes`` (``FactorOracleConstraint``) are converted into auxiliary CPDs during supported
inference workflows.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_pyomo_conin`` requires the values of ``A`` and ``B``
to stay fixed across the first two time steps:

.. testcode::

   import pyomo.environ as pyo
   from conin import pyomo_constraint_fn
   from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.examples.dynamic_bayesian_network import simple1_DDBN_conin

   base = simple1_DDBN_conin().pgm

   @pyomo_constraint_fn()
   def constraints(model):
       model.c = pyo.ConstraintList()
       model.c.add(model.V("A", 0, 0) == model.V("A", 1, 0))
       model.c.add(model.V("B", 0, 0) == model.V("B", 1, 0))

   constrained = ConstrainedDynamicDiscreteBayesianNetwork(
       base,
       constraints=constraints,
   )

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_toulbar2_conin`` expresses the same condition with
Toulbar2:

.. testcode::

   from conin import toulbar2_constraint_fn
   from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.examples.dynamic_bayesian_network import simple1_DDBN_conin

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
       constraints=constraints,
   )

Oracle constraints with node scoping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``simple1_DDBN_constrained_oracle_conin`` creates a constraint over a generated
set of nodes:

.. testcode::

   from conin import oracle_constraint_fn
   from conin.constrained_models import ConstrainedDynamicDiscreteBayesianNetwork
   from conin.examples.dynamic_bayesian_network import simple1_DDBN_conin

   base = simple1_DDBN_conin().pgm

   def nodes(data):
       for t in data.T:
           yield ("A", t)
           yield ("B", t)

   @oracle_constraint_fn(nodes=nodes)
   def constraints(states):
       return states["A", 0] == states["A", 1] and states["B", 0] == states["B", 1]

   constrained = ConstrainedDynamicDiscreteBayesianNetwork(
       base,
       constraints=constraints,
   )

Notes
-----

- ``cpds`` is required and must be non-empty, and at least one of ``states``
  and ``dynamic_states`` must be non-empty.
- ``dynamic_states`` lists the state space for time-indexed variables.
- ``t`` is a symbolic ``ExpressionVariable`` used when declaring repeated CPDs.
  Pass the same object that the CPDs use to the constructor; it is available
  afterwards as ``dbn.t``.
- See :doc:`model_conversion_io` for unrolling dynamic Bayesian networks into
  static Bayesian networks.
- Simpler models such as ``simple0_DDBN_conin`` and ``simple1_DDBN_conin`` are
  better documentation examples than the larger weather model.
