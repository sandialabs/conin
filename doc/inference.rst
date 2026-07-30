Inference
=========

This page illustrates the main inference wrappers on small Markov network and
hidden Markov model examples.

Markov network inference
------------------------

The ``ABC_conin`` example from ``conin.markov_network.examples`` is a compact
Markov network with three variables and pairwise interactions.

.. code-block:: python

   from conin.inference import (
       CFNInference,
       IntegerProgrammingInference,
       VariableEliminationInference,
   )
   from conin.markov_network.examples import ABC_conin

   example = ABC_conin()
   pgm = example.pgm

   cfn_results = CFNInference(pgm).map_query()
   ip_results = IntegerProgrammingInference(pgm).map_query(solver="glpk")
   ve_results = VariableEliminationInference(pgm).map_query()

   print(cfn_results.solution.states)
   print(ip_results.solution.states)
   print(ve_results.solution.states)

``CFNInference`` dispatches to the Toulbar2 backend, ``IntegerProgrammingInference``
creates a Pyomo optimization model, and ``VariableEliminationInference`` uses
pgmpy's variable elimination solver.

Hidden Markov model inference
-----------------------------

The ``create_hmm1`` example from ``conin.hidden_markov_model.tests.examples`` is
small enough to use throughout the documentation.

Viterbi and A* inference
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conin.inference import AStarInference, ViterbiInference
   from conin.hidden_markov_model.tests.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   a_star_results = AStarInference(hmm).map_query(evidence=observed)
   viterbi_results = ViterbiInference(hmm).map_query(evidence=observed)

   print(a_star_results.solution.states)
   print(viterbi_results.solution.states)

Both wrappers accept dense evidence lists. For HMM wrappers that also support a
dictionary form, a mapping such as ``{0: "o0", 1: "o0", 2: "o1"}`` can be used
when you want the returned hidden states keyed by time index.

Optimization and Toulbar2 inference on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dynamic-programming-style HMM wrappers use the ``DPGM_*`` classes.

.. code-block:: python

   from conin.inference import DPGM_CFNInference, DPGM_IntegerProgrammingInference
   from conin.hidden_markov_model.tests.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   cfn_results = DPGM_CFNInference(hmm).map_query(evidence=observed)
   ip_results = DPGM_IntegerProgrammingInference(hmm).map_query(
       evidence=observed,
       solver="glpk",
   )

   print(cfn_results.solution.states)
   print(ip_results.solution.states)

Variable elimination on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DPGM_VariableEliminationInference`` first converts the HMM into a dynamic
Bayesian network and then unrolls it into a static Bayesian network for pgmpy.

.. code-block:: python

   from conin.inference import DPGM_VariableEliminationInference
   from conin.hidden_markov_model.tests.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   results = DPGM_VariableEliminationInference(hmm).map_query(evidence=observed)
   print(results.solution.states)

Constrained HMM inference
-------------------------

The constrained examples from
``conin.hidden_markov_model.tests.examples`` can be used with the same wrappers.
For example, the Pyomo-constrained model ``create_chmm1_pyomo()`` works with
``DPGM_IntegerProgrammingInference``, and the Toulbar2-constrained model
``create_chmm1_toulbar2()`` works with ``DPGM_CFNInference``.

.. code-block:: python

   from conin.inference import DPGM_CFNInference, DPGM_IntegerProgrammingInference
   from conin.hidden_markov_model.tests.examples import (
       create_chmm1_pyomo,
       create_chmm1_toulbar2,
   )

   observed = ["o0"] * 15

   pyomo_hmm = create_chmm1_pyomo()
   pyomo_results = DPGM_IntegerProgrammingInference(pyomo_hmm).map_query(
       evidence=observed,
       solver="glpk",
   )

   toulbar2_hmm = create_chmm1_toulbar2()
   toulbar2_results = DPGM_CFNInference(toulbar2_hmm).map_query(evidence=observed)

   print(pyomo_results.solution.states)
   print(toulbar2_results.solution.states)

Notes
-----

- ``CFNInference`` and ``DPGM_CFNInference`` rely on Toulbar2.
- ``IntegerProgrammingInference`` and ``DPGM_IntegerProgrammingInference``
  require a Pyomo-compatible solver such as ``glpk``, ``highs``, or ``gurobi``.
- ``VariableEliminationInference`` and ``DPGM_VariableEliminationInference``
  require pgmpy.
- ``ViterbiInference`` and ``AStarInference`` are HMM-specific wrappers.
