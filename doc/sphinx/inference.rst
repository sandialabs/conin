Inference
=========

This page illustrates the main ``map_query`` inference entry point on small
Markov network and hidden Markov model examples.

Some methods require optional backend packages or external solvers. The
``viterbi`` and ``a_star`` HMM examples run with the base package dependencies.
See :doc:`backends` for backend-specific installation and solver requirements.

Markov network inference
------------------------

The ``ABC_conin`` example from ``conin.markov_network.examples`` is a compact
Markov network with three variables and pairwise interactions.

.. code-block:: python

   from conin.inference import map_query
   from conin.markov_network.examples import ABC_conin

   example = ABC_conin()
   pgm = example.pgm

   cfn_results = map_query(pgm, method="toulbar2")
   ip_results = map_query(pgm, method="integer_program", solver="glpk")
   ve_results = map_query(pgm, method="variable_elimination")

   print(cfn_results.solution.states)
   print(ip_results.solution.states)
   print(ve_results.solution.states)

The ``method`` argument selects the backend: ``toulbar2`` dispatches to the
Toulbar2 backend, ``integer_program`` creates a Pyomo optimization model, and
``variable_elimination`` uses pgmpy's variable elimination solver.

Hidden Markov model inference
-----------------------------

The ``create_hmm1`` example from ``conin.hidden_markov_model.examples`` is
small enough to use throughout the documentation.

Viterbi and A* inference
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from conin.inference import map_query
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   a_star_results = map_query(hmm, method="a_star", evidence=observed)
   viterbi_results = map_query(hmm, method="viterbi", evidence=observed)

   print(a_star_results.solution.states)
   print(viterbi_results.solution.states)

Both methods accept dense evidence lists. Methods that also support a dictionary
form can use a mapping such as ``{0: "o0", 1: "o0", 2: "o1"}`` when you want the
returned hidden states keyed by time index.

Optimization and Toulbar2 inference on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic Bayesian networks and HMMs use the same ``map_query`` function. The
input model type and selected ``method`` determine the backend dispatch.

.. code-block:: python

   from conin.inference import map_query
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   cfn_results = map_query(hmm, method="toulbar2", evidence=observed)
   ip_results = map_query(
       hmm,
       method="integer_program",
       evidence=observed,
       solver="glpk",
   )

   print(cfn_results.solution.states)
   print(ip_results.solution.states)

Variable elimination on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``variable_elimination`` method first converts the HMM into a dynamic
Bayesian network and then unrolls it into a static Bayesian network for pgmpy.

.. code-block:: python

   from conin.inference import map_query
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   results = map_query(hmm, method="variable_elimination", evidence=observed)
   print(results.solution.states)

Constrained HMM inference
-------------------------

The constrained examples from
``conin.hidden_markov_model.examples`` can be used with the same ``map_query``
entry point.
For example, the Pyomo-constrained model ``create_chmm1_pyomo()`` works with
``method="integer_program"``, and the Toulbar2-constrained model
``create_chmm1_toulbar2()`` works with ``method="toulbar2"``.

.. code-block:: python

   from conin.inference import map_query
   from conin.hidden_markov_model.examples import (
       create_chmm1_pyomo,
       create_chmm1_toulbar2,
   )

   observed = ["o0"] * 15

   pyomo_hmm = create_chmm1_pyomo()
   pyomo_results = map_query(
       pyomo_hmm,
       method="integer_program",
       evidence=observed,
       solver="glpk",
   )

   toulbar2_hmm = create_chmm1_toulbar2()
   toulbar2_results = map_query(
       toulbar2_hmm,
       method="toulbar2",
       evidence=observed,
   )

   print(pyomo_results.solution.states)
   print(toulbar2_results.solution.states)

Notes
-----

Backend requirements for these methods are maintained in :doc:`backends`.
