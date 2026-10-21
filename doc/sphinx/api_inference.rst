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
