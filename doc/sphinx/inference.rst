Inference
=========

This page illustrates the main ``map_query`` inference entry point on small
Markov network and hidden Markov model examples.

Some methods require optional backend packages or external solvers. The
``viterbi`` and ``a_star`` HMM examples run with the base package dependencies.
See :doc:`backends` for backend-specific installation and solver requirements.

Markov network inference
------------------------

The ``ABC_conin`` example from ``conin.examples.markov_network`` is a compact
Markov network with three variables and pairwise interactions.

.. testcode::
   :skipif: not pgmpy_available or not toulbar2_available or not mip_solver_available

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.markov_network import ABC_conin

   example = ABC_conin()
   pgm = example.pgm

   cfn_results = map_query(pgm, method="toulbar2")
   ip_results = map_query(pgm, method="integer_program", solver=default_mip_solver)
   ve_results = map_query(pgm, method="variable_elimination")

   print(sorted(cfn_results.solution.states.items()))
   assert ip_results.solution.states == cfn_results.solution.states
   assert ve_results.solution.states == cfn_results.solution.states

.. testoutput::
   :skipif: not pgmpy_available or not toulbar2_available or not mip_solver_available

   [('A', 2), ('B', 2), ('C', 1)]

The ``method`` argument selects the backend: ``toulbar2`` dispatches to the
Toulbar2 backend, ``integer_program`` creates a Pyomo optimization model, and
``variable_elimination`` uses pgmpy's variable elimination solver.

Hidden Markov model inference
-----------------------------

The ``create_hmm1`` example from ``conin.examples.hidden_markov_model`` is
small enough to use throughout the documentation.

Viterbi and A* inference
^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode::

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   a_star_results = map_query(hmm, method="a_star", evidence=observed)
   viterbi_results = map_query(hmm, method="viterbi", evidence=observed)

   print(a_star_results.solution.states)
   print(viterbi_results.solution.states)

.. testoutput::

   ['h0', 'h0', 'h0', 'h0', 'h0']
   ['h0', 'h0', 'h0', 'h0', 'h0']

Both methods accept the observations as a list or tuple, or as a dictionary
keyed by every time ``0..T-1``, such as ``{0: "o0", 1: "o0", 2: "o1"}``, which
returns the hidden states keyed by time index. HMM queries need the
observations: other evidence (including a dictionary with missing times)
raises ``InvalidInputError``. The integer-program, toulbar2 and
variable-elimination methods infer over the times ``start..stop``, with
``stop`` defaulting to the last observation.

Optimization and Toulbar2 inference on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic Bayesian networks and HMMs use the same ``map_query`` function. The
input model type and selected ``method`` determine the backend dispatch.

.. testcode::
   :skipif: not toulbar2_available or not mip_solver_available

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   cfn_results = map_query(hmm, method="toulbar2", evidence=observed)
   ip_results = map_query(
       hmm,
       method="integer_program",
       evidence=observed,
       solver=default_mip_solver,
   )

   print(cfn_results.solution.states)
   print(ip_results.solution.states)

.. testoutput::
   :skipif: not toulbar2_available or not mip_solver_available

   ['h0', 'h0', 'h0', 'h0', 'h0']
   ['h0', 'h0', 'h0', 'h0', 'h0']

Variable elimination on HMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``variable_elimination`` method first converts the HMM into a dynamic
Bayesian network and then unrolls it into a static Bayesian network for pgmpy.

.. testcode::
   :skipif: not pgmpy_available

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1", "o0", "o0"]

   results = map_query(hmm, method="variable_elimination", evidence=observed)
   print(results.solution.states)

.. testoutput::
   :skipif: not pgmpy_available

   ['h0', 'h0', 'h0', 'h0', 'h0']

Constrained HMM inference
-------------------------

The constrained examples from
``conin.examples.hidden_markov_model`` can be used with the same ``map_query``
entry point.
For example, the Pyomo-constrained model ``create_chmm1_pyomo()`` works with
``method="integer_program"``, the Toulbar2-constrained model
``create_chmm1_toulbar2()`` works with ``method="toulbar2"``, and the oracle and
recursive models ``create_chmm1_oracle()`` and ``create_chmm1_recursive()``
work with ``method="a_star"``. HMMs with MVR constraints work with
``method="mvr_viterbi"`` (requires torch).

.. testcode::
   :skipif: not toulbar2_available or not mip_solver_available

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.hidden_markov_model import (
       create_chmm1_pyomo,
       create_chmm1_toulbar2,
   )

   observed = ["o0"] * 15

   pyomo_hmm = create_chmm1_pyomo()
   pyomo_results = map_query(
       pyomo_hmm,
       method="integer_program",
       evidence=observed,
       solver=default_mip_solver,
   )

   toulbar2_hmm = create_chmm1_toulbar2()
   toulbar2_results = map_query(
       toulbar2_hmm,
       method="toulbar2",
       evidence=observed,
   )

   print(pyomo_results.solution.states)
   print(toulbar2_results.solution.states)

.. testoutput::
   :skipif: not toulbar2_available or not mip_solver_available

   ['h1', 'h1', 'h1', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0']
   ['h1', 'h1', 'h1', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0', 'h0']

Notes
-----

Backend requirements for these methods are maintained in :doc:`backends`.
