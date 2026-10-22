Inference API
=============

This page documents the public inference entry point. Use
``conin.inference.map_query`` with a ``method`` argument to select the backend
algorithm. ``map_query`` turns the model into a constrained model (a bare model
has no constraints; pgmpy models are converted), builds the representation that
the method needs and dispatches on its type, so the same call shape works
across supported Markov networks, Bayesian networks, dynamic Bayesian
networks, hidden Markov models, and constrained variants when the selected
backend supports them. A method that does not support the model type or its
constraint family raises ``InvalidInputError``.

.. autofunction:: conin.inference.map_query

Available Methods
-----------------

``integer_program``
   Uses the Pyomo optimization backend. ``ip_formulation="network_flow"``
   selects the network-flow formulation for hidden Markov models.

``toulbar2``
   Uses the Toulbar2 cost-function-network backend.

``variable_elimination``
   Uses pgmpy's variable-elimination backend.

``a_star``
   Uses A* search for supported hidden Markov model inputs (oracle
   constraints, or recursive A* for recursive constraints). When no feasible
   sequence is found, ``solution`` is ``None``.

``viterbi``
   Uses the Viterbi algorithm for unconstrained hidden Markov models.

``mvr_viterbi``
   Uses the torch MVR Viterbi algorithm for hidden Markov models with MVR
   constraints (requires torch). The solution also has ``log_likelihood`` and
   ``augmented`` (the decoded augmented path).

Backend functions
-----------------

``map_query`` calls one of the functions below. They can also be called
directly; each takes a bare or constrained model and builds the representation
it needs.

Unconstrained, oracle and recursive search (HMMs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conin.inference.unconstrained.viterbi

.. autofunction:: conin.inference.oracle.a_star

.. autofunction:: conin.inference.recursive.recursive_a_star

Pyomo (integer programming)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conin.inference.pyomo.inference_pyomo_map_query_MN

.. autofunction:: conin.inference.pyomo.inference_pyomo_map_query_BN

.. autofunction:: conin.inference.pyomo.inference_pyomo_map_query_DDBN

.. autofunction:: conin.inference.pyomo.inference_pyomo_map_query_HMM

.. autofunction:: conin.inference.pyomo.lp_inference

.. autofunction:: conin.inference.pyomo.ip_inference

.. autofunction:: conin.inference.pyomo.solve_pyomo_map_query_model

Toulbar2
^^^^^^^^

.. autofunction:: conin.inference.toulbar2.inference_toulbar2_map_query_MN

.. autofunction:: conin.inference.toulbar2.inference_toulbar2_map_query_BN

.. autofunction:: conin.inference.toulbar2.inference_toulbar2_map_query_DDBN

.. autofunction:: conin.inference.toulbar2.inference_toulbar2_map_query_HMM

.. autofunction:: conin.inference.toulbar2.solve_toulbar2_map_query_model

Variable elimination (pgmpy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conin.inference.variable_elimination.inference_variable_elimination_map_query_MN

.. autofunction:: conin.inference.variable_elimination.inference_variable_elimination_map_query_BN

.. autofunction:: conin.inference.variable_elimination.inference_variable_elimination_map_query_DDBN

.. autofunction:: conin.inference.variable_elimination.inference_variable_elimination_map_query_HMM

.. autofunction:: conin.inference.variable_elimination.solve_variable_elimination_map_query_model

MVR (torch)
^^^^^^^^^^^

These need torch, which is imported on first use.

.. autofunction:: conin.inference.mvr.viterbi.viterbi_torch_mvr_chmm

.. autofunction:: conin.inference.mvr.marginal_map.marginal_map_torch_mvr_chmm

Satisfaction probability and first satisfaction time of MVR constraints:

.. autofunction:: conin.inference.mvr.sat_prob.sat_prob_torch_mvr_chmm

.. autofunction:: conin.inference.mvr.sat_time.sat_time_torch_mvr_chmm
