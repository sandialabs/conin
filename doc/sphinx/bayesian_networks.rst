Creating a Discrete Bayesian Network
====================================

This page describes how to create a ``DiscreteBayesianNetwork`` instance using
``DiscreteCPD`` objects.

Conditional Probability Distributions
-------------------------------------

``DiscreteCPD`` stores the conditional distribution for one node. Root-node CPDs
can be supplied as a list ordered by the node's state list, while CPDs with
parents can be supplied as a dictionary keyed by parent assignments.

.. testcode::

   from conin import DiscreteBayesianNetwork, DiscreteCPD

   cpd_a = DiscreteCPD(node="A", values=[0.9, 0.1])
   cpd_b = DiscreteCPD(
       node="B",
       parents=["A"],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )

   pgm = DiscreteBayesianNetwork(
       states={"A": [0, 1], "B": [0, 1]},
       cpds=[cpd_a, cpd_b],
   )

When CPDs are attached to a ``DiscreteBayesianNetwork``, list-valued entries are
normalized into dictionaries keyed by the model's states.

Bayesian networks
-----------------

The ``simple1_BN_conin`` example is a good starting point because it has only
two nodes and one conditional dependency.

.. testcode::

   from conin import DiscreteBayesianNetwork, DiscreteCPD

   cpd_a = DiscreteCPD(node="A", values=[0.9, 0.1])
   cpd_b = DiscreteCPD(
       node="B",
       parents=["A"],
       values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   )

   pgm = DiscreteBayesianNetwork(
       states={"A": [0, 1], "B": [0, 1]},
       cpds=[cpd_a, cpd_b],
   )

A larger example is ``cancer1_BN_conin``, which uses explicit state names and a
node with two parents:

.. testcode::

   from conin import DiscreteBayesianNetwork, DiscreteCPD

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

   cancer = DiscreteBayesianNetwork(
       states={
           "Cancer": [0, 1],
           "Dyspnoea": [0, 1],
           "Pollution": [0, 1],
           "Smoker": [0, 1],
           "Xray": [0, 1],
       },
       cpds=[
           cpd_pollution,
           cpd_smoker,
           cpd_cancer,
           cpd_xray,
           cpd_dyspnoea,
       ],
   )

Constrained Bayesian networks
-----------------------------

``ConstrainedDiscreteBayesianNetwork`` uses the same constraint decorators as
other model families. The examples in ``conin.examples.bayesian_network`` apply
constraints to the cancer network.

The Pyomo and Toulbar2 snippets below require their respective solver backends.
Oracle constraints with ``nodes`` (``FactorOracleConstraint``) are converted into
indicator factors (``0`` on infeasible assignments) by the variable-elimination
backend.

Pyomo constraints
^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_pyomo_conin`` prevents ``Dyspnoea`` and ``Xray`` from
choosing the same state:

.. testcode::

   import pyomo.environ as pyo
   from conin import pyomo_constraint_fn, ConstrainedDiscreteBayesianNetwork
   from conin.examples.bayesian_network import cancer1_BN_conin

   base = cancer1_BN_conin().pgm

   @pyomo_constraint_fn()
   def constraints(model):
       model.c = pyo.ConstraintList()
       model.c.add(model.V("Dyspnoea", 1) + model.V("Xray", 1) <= 1)
       model.c.add(model.V("Dyspnoea", 0) + model.V("Xray", 0) <= 1)

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=constraints)

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_toulbar2_conin`` uses the Toulbar2 linear-constraint
interface:

.. testcode::

   from conin import toulbar2_constraint_fn, ConstrainedDiscreteBayesianNetwork
   from conin.examples.bayesian_network import cancer1_BN_conin

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

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=constraints)

Oracle constraints with node scoping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cancer1_BN_constrained_oracle_conin`` creates a binary constraint as an
auxiliary CPD:

.. testcode::

   from conin import oracle_constraint_fn, ConstrainedDiscreteBayesianNetwork
   from conin.examples.bayesian_network import cancer1_BN_conin

   base = cancer1_BN_conin().pgm

   @oracle_constraint_fn(nodes=["Dyspnoea", "Xray"])
   def constraints(states):
       return states["Dyspnoea"] != states["Xray"]

   constrained = ConstrainedDiscreteBayesianNetwork(base, constraints=constraints)

Notes
-----

- ``cpds`` is required and must be non-empty. Models are immutable after
  construction; use ``with_cpd`` to get a new model with an extra CPD (the
  new model must be valid too).
- ``states`` defines the state names used by every CPD. It is optional: the
  states of nodes that are not listed are inferred from their CPDs (which
  then need dictionary values).
- ``DiscreteCPD`` supports both dictionary-based and list-based probability
  declarations.
- The constructor validates the network: the CPDs must
  be consistent with the network structure and state definitions, every
  node has exactly one CPD, every distribution (for every parent state)
  sums to 1, a list of values has one value per state (and parent state),
  and the CPDs have no cycle. It raises ``InvalidInputError`` if not, so a
  network that exists is valid.
- See :doc:`model_conversion_io` for converting Bayesian networks to Markov
  networks and saving/loading UAI files.
- For introductory work, ``simple1_BN_conin`` is the easiest example to adapt.
