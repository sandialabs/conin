Quickstart and Installation
===========================

This page shows the quickest path from installation to a small CONIN Bayesian
network. It uses only the base package dependencies.

Installation
------------

For local development from a repository checkout, install CONIN in editable
mode:

.. code-block:: shell

   python -m pip install -e .

To build the documentation locally, install the documentation extra:

.. code-block:: shell

   python -m pip install -e .[docs]

To run the test suite, install the test extra:

.. code-block:: shell

   python -m pip install -e .[test]

Optional integrations and external solver requirements are summarized in
:doc:`backends`.

Build a Bayesian Network
------------------------

The following example creates a two-node Bayesian network with one root node
``A`` and one child node ``B``.

.. doctest:: quickstart

   >>> from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
   >>> cpd_a = DiscreteCPD(node="A", values=[0.9, 0.1])
   >>> cpd_b = DiscreteCPD(
   ...     node="B",
   ...     parents=["A"],
   ...     values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   ... )
   >>> pgm = DiscreteBayesianNetwork(states={"A": [0, 1], "B": [0, 1]}, cpds=[cpd_a, cpd_b])
   >>> pgm.check_model()
   >>> pgm.nodes
   ['A', 'B']
   >>> pgm.edges
   [('A', 'B')]

Evaluate an Assignment
----------------------

``conin.scoring.log_probability`` returns the log-probability of a total
assignment of a Bayesian network, a dynamic Bayesian network or an HMM.
``conin.scoring.log_potential`` also accepts Markov networks (where it returns
the unnormalised log-probability) and constrained models (where it returns
``-inf`` if a constraint is violated). ``conin.scoring.log_probability_query``
computes marginal and conditional log-probabilities, also under constraints.
See :doc:`model_conversion_io`.

.. doctest:: quickstart

   >>> from conin.scoring import log_probability
   >>> value = log_probability(pgm, {"A": 0, "B": 1})
   >>> round(value, 6)
   -0.328504

Run Inference
-------------

For MAP inference on Bayesian networks, use the high-level ``map_query`` entry
point with the desired backend method. Backend-specific requirements are listed
in :doc:`backends`.

For example, with ``pgmpy`` installed:

.. testcode:: quickstart
   :skipif: not pgmpy_available

   from conin.inference import map_query

   result = map_query(pgm, method="variable_elimination")
   print(sorted(result.solution.states.items()))

.. testoutput:: quickstart
   :skipif: not pgmpy_available

   [('A', 0), ('B', 1)]

Next Steps
----------

- See :doc:`probabilistic_graphical_models` for model-building guides.
- See :doc:`inference` for inference methods.
- See :doc:`examples` for selected example factories.
- See :doc:`model_conversion_io` for model conversion and file I/O.
- See :doc:`hmm_learning` for HMM parameter learning.
