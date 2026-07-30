Creating a Discrete Bayesian Network
====================================

This page describes how to create a ``DiscreteBayesianNetwork`` instance using
``DiscreteCPD`` objects.

Conditional Probability Distributions
-------------------------------------

TODO

Bayesian networks
-----------------

The ``simple1_BN_conin`` example is a good starting point because it has only
two nodes and one conditional dependency.

.. code-block:: python

   from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD

   pgm = DiscreteBayesianNetwork()
   pgm.states = {"A": [0, 1], "B": [0, 1]}

   cpd_a = DiscreteCPD(node="A", values=[0.9, 0.1])
   cpd_b = DiscreteCPD(
       node="B",
       parents=["A"],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )

   pgm.cpds = [cpd_a, cpd_b]
   pgm.check_model()

A larger example is ``cancer1_BN_conin``, which uses explicit state names and a
node with two parents:

.. code-block:: python

   from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD

   cancer = DiscreteBayesianNetwork()
   cancer.states = {
       "Cancer": [0, 1],
       "Dyspnoea": [0, 1],
       "Pollution": [0, 1],
       "Smoker": [0, 1],
       "Xray": [0, 1],
   }

   cpd_pollution = DiscreteCPD(node="Pollution", values=[0.9, 0.1])
   cpd_smoker = DiscreteCPD(node="Smoker", values=[0.3, 0.7])
   cpd_cancer = DiscreteCPD(
       node="Cancer",
       parents=["Smoker", "Pollution"],
       values={
           (0, 0): [0.03, 0.97],
           (0, 1): [0.05, 0.95],
           (1, 0): [0.001, 0.999],
           (1, 1): [0.02, 0.98],
       },
   )
   cpd_xray = DiscreteCPD(
       node="Xray",
       parents=["Cancer"],
       values={0: [0.9, 0.1], 1: [0.2, 0.8]},
   )
   cpd_dyspnoea = DiscreteCPD(
       node="Dyspnoea",
       parents=["Cancer"],
       values={0: [0.65, 0.35], 1: [0.3, 0.7]},
   )

   cancer.cpds = [
       cpd_pollution,
       cpd_smoker,
       cpd_cancer,
       cpd_xray,
       cpd_dyspnoea,
   ]
   cancer.check_model()

Constrained Bayesian networks
-----------------------------

``ConstrainedDiscreteBayesianNetwork`` uses the same constraint decorators as
other model families. The examples in ``conin.bayesian_network.examples`` apply
constraints to the cancer network.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_pyomo_conin`` prevents ``Dyspnoea`` and ``Xray`` from
choosing the same state:

.. code-block:: python

   import pyomo.environ as pyo
   from conin import pyomo_constraint_fn
   from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
   from conin.bayesian_network.examples import cancer1_BN_conin

   base = cancer1_BN_conin().pgm

   @pyomo_constraint_fn()
   def constraints(model):
       model.c = pyo.ConstraintList()
       model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
       model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=[constraints])

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_toulbar2_conin`` uses the Toulbar2 linear-constraint
interface:

.. code-block:: python

   from conin import toulbar2_constraint_fn
   from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
   from conin.bayesian_network.examples import cancer1_BN_conin

   base = cancer1_BN_conin().pgm

   @toulbar2_constraint_fn()
   def constraints(model):
       model.AddGeneralizedLinearConstraint(
           [model.V("Dyspnoea", 1), model.V("Xray", 1)],
           "<=",
           1,
       )
       model.AddGeneralizedLinearConstraint(
           [model.V("Dyspnoea", 0), model.V("Xray", 0)],
           "<=",
           1,
       )

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=[constraints])

Factor constraints
^^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_factor_conin`` creates a binary constraint as an
auxiliary CPD:

.. code-block:: python

   from conin import factor_constraint_fn
   from conin.bayesian_network import ConstrainedDiscreteBayesianNetwork
   from conin.bayesian_network.examples import cancer1_BN_conin

   base = cancer1_BN_conin().pgm

   @factor_constraint_fn(nodes=["Dyspnoea", "Xray"])
   def constraints(states):
       return states["Dyspnoea"] != states["Xray"]

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=[constraints])

Notes
-----

- ``states`` defines the state names used by every CPD.
- ``DiscreteCPD`` supports both dictionary-based and list-based probability
  declarations.
- ``check_model()`` verifies that the CPDs are consistent with the network
  structure and state definitions.
- For introductory work, ``simple1_BN_conin`` is the easiest example to adapt.
