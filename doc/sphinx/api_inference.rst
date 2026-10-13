Inference API
=============

This page documents the public inference entry point. Use
``conin.inference.map_query`` with a ``method`` argument to select the backend
algorithm. The implementation dispatches on the model type, so the same call
shape works across supported Markov networks, Bayesian networks, dynamic
Bayesian networks, hidden Markov models, and constrained variants when the
selected backend supports them.

.. autofunction:: conin.inference.map_query

Available Methods
-----------------

``integer_program``
   Uses the Pyomo optimization backend.

``toulbar2``
   Uses the Toulbar2 cost-function-network backend.

``variable_elimination``
   Uses pgmpy's variable-elimination backend.

``a_star``
   Uses A* search for supported hidden Markov model inputs.

``viterbi``
   Uses the Viterbi algorithm for supported hidden Markov model inputs.
