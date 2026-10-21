Creating a Hidden Markov Model
==============================

This page describes how to create a ``HiddenMarkovModel`` instance and how to
create constrained hidden Markov models.

Hidden Markov models
--------------------

The ``create_hmm1`` example is a compact starting point. It defines start,
transition, and emission probabilities and then passes them to the
``HiddenMarkovModel`` constructor.

.. code-block:: python

   from conin.graphical_models import HiddenMarkovModel

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

   hmm = HiddenMarkovModel(
       start_probs=start_probs,
       transition_probs=transition_probs,
       emission_probs=emission_probs,
   )
   hmm.set_seed(0)

The hidden and observed state labels are available from ``hmm.hidden_states``
and ``hmm.observed_states``. Internally, states are indexed in sorted label
order.

Random HMMs
-----------

Use ``random_hmm`` to create a reproducible HMM with randomly generated start,
transition, and emission probabilities over a specified state space.

.. code-block:: python

   from conin.graphical_models import random_hmm

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

   from conin.graphical_models import HiddenMarkovModel
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()
   hmm.write_to_file("hmm.json")

   loaded = HiddenMarkovModel.read_from_file("hmm.json")

   assert loaded.get_start_probs() == hmm.get_start_probs()
   assert loaded.get_transition_probs() == hmm.get_transition_probs()
   assert loaded.get_emission_probs() == hmm.get_emission_probs()

Constrained hidden Markov models
--------------------------------

``ConstrainedHiddenMarkovModel`` wraps a base HMM and a list of
constraints. The public examples include oracle, Pyomo, and Toulbar2 constraints.
Oracle constraints with ``nodes`` (``FactorOracleConstraint``) generate
auxiliary factors for variable-elimination inference.

Constrained models are immutable: ``ConstrainedHiddenMarkovModel(hmm,
constraints, data=None)`` stores the HMM as ``.model`` and the constraints as a
tuple in ``.constraints``. All constraints must belong to one constraint family,
which is available as ``.constraint_family`` (``None`` when there are no
constraints); mixing families raises ``InvalidInputError``. Application data
used by constraint functions and MVR factories is passed to the constructor
with ``data=``.

Backend representations are built by ``representation(kind)``; each call
builds a new one. HMMs register these kinds: ``"oracle"`` (feasibility checks
for A* and rejection sampling; no constraints or oracle constraints),
``"recursive"`` (for recursive A*; no constraints or recursive constraints),
``"factor"`` (variable elimination; no constraints, factor-oracle or algebraic
constraints), ``"pyomo"`` (the integer program of the
unrolled model; no constraints, Pyomo or algebraic constraints),
``"pyomo_network_flow"`` (the network-flow integer program; no constraints,
Pyomo or algebraic constraints), ``"toulbar2"`` (the cost function network of
the unrolled model; no constraints, Toulbar2 or algebraic constraints) and
``"mvr"`` (no constraints or MVR constraints; the built MVRs, aligned to the
HMM, are ``chmm.representation("mvr").mvrs``). The torch MVR algorithms in
``conin.inference.mvr``, ``conin.learning.mvr``, ``conin.sampling.mvr`` and
``conin.queries.mvr`` take a ``ConstrainedHiddenMarkovModel`` with MVR
constraints; see the notebooks in ``doc/notebooks/MVR``.

Oracle constraints
^^^^^^^^^^^^^^^^^^

``create_chmm1_oracle`` uses two oracle constraints to restrict the number of
``"h0"`` states in the hidden sequence.

.. code-block:: python

   from conin.graphical_models import HiddenMarkovModel
   from conin.constrained_models import ConstrainedHiddenMarkovModel
   from conin.constraints import OracleConstraint
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()

   num_zeros_greater_than_nine = OracleConstraint(
       func=lambda states: sum(1 for v in states.values() if v == "h0") > 9,
       partial_func=lambda T, states: T - len(states) + sum(1 for v in states.values() if v == "h0") >= 10,
   )
   num_zeros_less_than_thirteen = OracleConstraint(
       func=lambda states: sum(1 for v in states.values() if v == "h0") < 13,
       partial_func=lambda T, states: sum(1 for v in states.values() if v == "h0") < 13,
   )

   chmm = ConstrainedHiddenMarkovModel(
       hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

Pyomo constraints
^^^^^^^^^^^^^^^^^

``create_chmm1_pyomo`` expresses the same condition algebraically:

.. code-block:: python

   import pyomo.environ as pe
   import conin
   from conin.constrained_models import ConstrainedHiddenMarkovModel
   from conin.examples.hidden_markov_model import create_hmm1

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
       hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

Toulbar2 constraints
^^^^^^^^^^^^^^^^^^^^

``create_chmm1_toulbar2`` uses the Toulbar2 linear-constraint interface:

.. code-block:: python

   from conin import toulbar2_constraint_fn
   from conin.constrained_models import ConstrainedHiddenMarkovModel
   from conin.examples.hidden_markov_model import create_hmm1

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
       hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

``map_query(..., method="toulbar2")`` applies these constraints when building
the Toulbar2 model.

Oracle constraints with node scoping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``create_chmm1_oracle_ve`` adds the same logic through generated factors:

.. code-block:: python

   from conin import oracle_constraint_fn
   from conin.constrained_models import ConstrainedHiddenMarkovModel
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()

   def nodes(data):
       for t in data.hmm.T:
           yield ("H", t)

   @oracle_constraint_fn(nodes=nodes)
   def num_zeros_greater_than_nine(states, data):
       num = sum(1 for _, value in states.items() if value == "h0")
       return num >= 10

   @oracle_constraint_fn(nodes=nodes)
   def num_zeros_less_than_thirteen(states, data):
       num = sum(1 for _, value in states.items() if value == "h0")
       return num <= 12

   chmm = ConstrainedHiddenMarkovModel(
       hmm,
       constraints=[num_zeros_greater_than_nine, num_zeros_less_than_thirteen],
   )

``oracle_constraint_fn(nodes=...)`` creates a ``FactorOracleConstraint``, which
``map_query(..., method="variable_elimination")`` expands into auxiliary factors.

Recursive constraints
^^^^^^^^^^^^^^^^^^^^^

A ``RecursiveConstraint`` summarises a prefix of the hidden sequence in a
hashable state, updated one hidden state at a time. ``recursive_a_star`` merges
prefixes with the same state, which can make it much faster than A* with an
oracle constraint. ``create_chmm1_recursive`` expresses the condition of
``create_chmm1_oracle`` this way:

.. testcode::

   from conin.constrained_models import ConstrainedHiddenMarkovModel
   from conin.constraints import RecursiveConstraint
   from conin.examples.hidden_markov_model import create_hmm1
   from conin.inference.recursive import recursive_a_star

   lb, ub = 10, 12
   num_zeros = RecursiveConstraint(
       initial=lambda h: 1 if h == "h0" else 0,
       update=lambda n, h: n + 1 if h == "h0" else n,
       feasible=lambda n: lb <= n <= ub,
       feasible_partial=lambda n, t, T: n + (T - t) >= lb and n <= ub,
   )
   chmm = ConstrainedHiddenMarkovModel(create_hmm1(), num_zeros)

   results = recursive_a_star(hmm=chmm, observed=["o0"] * 15)
   print(results.solutions[0].hidden.count("h0"))

.. testoutput::

   12

``map_query(..., method="a_star")`` does not use recursive constraints yet; call
``recursive_a_star`` directly.

Notes
-----

- The ``HiddenMarkovModel`` constructor requires non-empty ``start_probs``,
  ``transition_probs``, and ``emission_probs`` dictionaries keyed by hidden and
  observed state labels. Models are immutable after construction.
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
