Creating a Hidden Markov Model
==============================

This page describes how to create a ``HiddenMarkovModel`` instance and how to
create constrained hidden Markov models.

Hidden Markov models
--------------------

The ``create_hmm1`` example is a compact starting point. It defines start,
transition, and emission probabilities and then loads them into a
``HiddenMarkovModel``.

.. code-block:: python

   from conin.hidden_markov_model import HiddenMarkovModel

   start_probs = {"h0": 0.4, "h1": 0.6}
   transition_probs = {
       ("h0", "h0"): 0.9,
       ("h0", "h1"): 0.1,
       ("h1", "h0"): 0.2,
       ("h1", "h1"): 0.8,
   }
   emission_probs = {
       ("h0", "o0"): 0.7,
       ("h0", "o1"): 0.3,
       ("h1", "o0"): 0.4,
       ("h1", "o1"): 0.6,
   }

   hmm = HiddenMarkovModel()
   hmm.load_model(
       start_probs=start_probs,
       transition_probs=transition_probs,
       emission_probs=emission_probs,
   )
   hmm.set_seed(0)

After ``load_model()``, the hidden and observed state labels are available from
``hmm.hidden_states`` and ``hmm.observed_states``.

Random HMMs
-----------

Use ``random_hmm`` to create a reproducible HMM with randomly generated start,
transition, and emission probabilities over a specified state space.

.. code-block:: python

   from conin.hidden_markov_model import random_hmm

   hmm = random_hmm(
       hidden_states=["h0", "h1"],
       observed_states=["o0", "o1"],
       seed=10,
   )

   assert hmm.hidden_states == ["h0", "h1"]
   assert hmm.observed_states == ["o0", "o1"]
   assert abs(sum(hmm.get_start_probs().values()) - 1.0) < 1e-12

JSON I/O
--------

``HiddenMarkovModel`` can write its start, transition, and emission
probabilities to a JSON file and load them back later. This HMM-specific JSON
format is separate from the UAI file support described in
:doc:`model_conversion_io`.

.. code-block:: python

   from conin.hidden_markov_model import HiddenMarkovModel
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   hmm.write_to_file("hmm.json")

   loaded = HiddenMarkovModel()
   loaded.read_from_file("hmm.json")

   assert loaded.get_start_probs() == hmm.get_start_probs()
   assert loaded.get_transition_probs() == hmm.get_transition_probs()
   assert loaded.get_emission_probs() == hmm.get_emission_probs()

Constrained hidden Markov models
--------------------------------

``ConstrainedHiddenMarkovModel`` wraps a base HMM and a list of constraint
functors. The public examples include oracle, factor, Pyomo, and Toulbar2
constraints.

``initialize_chmm()`` constructs an internal constrained HMM object for oracle,
Pyomo, and MVR constraints. Factor and Toulbar2 constraints are consumed by the
corresponding inference wrappers instead of initializing a standalone CHMM
object directly.

Oracle constraints
^^^^^^^^^^^^^^^^^^

``create_chmm1_oracle`` uses two oracle constraints to restrict the number of
``"h0"`` states in the hidden sequence.

.. code-block:: python

   from conin.hidden_markov_model import (
       HiddenMarkovModel,
       ConstrainedHiddenMarkovModel,
   )
   from conin.constraint import OracleConstraint
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()

   num_zeros_greater_than_nine = OracleConstraint(
       func=lambda seq: seq.count("h0") > 9,
       partial_func=lambda T, seq: T - len(seq) + seq.count("h0") >= 10,
   )
   num_zeros_less_than_thirteen = OracleConstraint(
       func=lambda seq: seq.count("h0") < 13,
       partial_func=lambda T, seq: seq.count("h0") < 13,
   )

   chmm = ConstrainedHiddenMarkovModel(
       hmm=hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )
   chmm.initialize_chmm()

Pyomo constraints
^^^^^^^^^^^^^^^^^

``create_chmm1_pyomo`` expresses the same condition algebraically:

.. code-block:: python

   import pyomo.environ as pe
   import conin
   from conin.hidden_markov_model import ConstrainedHiddenMarkovModel
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()

   @conin.pyomo_constraint_fn()
   def num_zeros_greater_than_nine(model, data):
       model.h0_lower = pe.Constraint(
           expr=sum(model.V("H", t, "h0") for t in data.hmm.T) >= 10
       )

   @conin.pyomo_constraint_fn()
   def num_zeros_less_than_thirteen(model, data):
       model.h0_upper = pe.Constraint(
           expr=sum(model.V("H", t, "h0") for t in data.hmm.T) <= 12
       )

   chmm = ConstrainedHiddenMarkovModel(
       hmm=hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )
   chmm.initialize_chmm()

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``create_chmm1_toulbar2`` uses the Toulbar2 linear-constraint interface:

.. code-block:: python

   from conin import toulbar2_constraint_fn
   from conin.hidden_markov_model import ConstrainedHiddenMarkovModel
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()

   @toulbar2_constraint_fn()
   def num_zeros_greater_than_nine(model, data):
       model.AddGeneralizedLinearConstraint(
           [model.V("H", t, "h0") for t in data.hmm.T],
           ">=",
           10,
       )

   @toulbar2_constraint_fn()
   def num_zeros_less_than_thirteen(model, data):
       model.AddGeneralizedLinearConstraint(
           [model.V("H", t, "h0") for t in data.hmm.T],
           "<=",
           12,
       )

   chmm = ConstrainedHiddenMarkovModel(
       hmm=hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

The ``DPGM_CFNInference`` wrapper applies these constraints when building the
Toulbar2 model.

Factor constraints
^^^^^^^^^^^^^^^^^^

``create_chmm1_factor`` adds the same logic through generated factors:

.. code-block:: python

   from conin import factor_constraint_fn
   from conin.hidden_markov_model import ConstrainedHiddenMarkovModel
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()

   def nodes(data):
       for t in data.hmm.T:
           yield ("H", t)

   @factor_constraint_fn(nodes=nodes)
   def num_zeros_greater_than_nine(states, data):
       num = sum(1 for _, value in states.items() if value == "h0")
       return num >= 10

   @factor_constraint_fn(nodes=nodes)
   def num_zeros_less_than_thirteen(states, data):
       num = sum(1 for _, value in states.items() if value == "h0")
       return num <= 12

   chmm = ConstrainedHiddenMarkovModel(
       hmm=hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

The ``DPGM_VariableEliminationInference`` wrapper can consume factor
constraints by expanding them into auxiliary factors.

Notes
-----

- ``HiddenMarkovModel.load_model()`` accepts dictionaries keyed by hidden and
  observed state labels.
- ``random_hmm`` is useful for tests, examples, and experiments where exact
  probabilities are less important than a valid model structure.
- ``write_to_file`` and ``read_from_file`` provide HMM-specific JSON
  serialization for model parameters.
- See :doc:`model_conversion_io` for converting HMMs to dynamic Bayesian
  networks.
- See :doc:`hmm_learning` for estimating HMM parameters from labeled
  simulations.
- ``ConstrainedHiddenMarkovModel`` requires all constraints in a model to use
  the same constraint family.
- ``create_hmm1`` and ``create_chmm1_*`` are the simplest tested examples for
  documentation and experimentation.
