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

   >>> from conin.bayesian_network import DiscreteBayesianNetwork, DiscreteCPD
   >>> pgm = DiscreteBayesianNetwork()
   >>> pgm.states = {"A": [0, 1], "B": [0, 1]}
   >>> cpd_a = DiscreteCPD(node="A", values=[0.9, 0.1])
   >>> cpd_b = DiscreteCPD(
   ...     node="B",
   ...     parents=["A"],
   ...     values={0: [0.2, 0.8], 1: [0.9, 0.1]},
   ... )
   >>> pgm.cpds = [cpd_a, cpd_b]
   >>> pgm.check_model()
   >>> pgm.nodes
   ['A', 'B']
   >>> pgm.edges
   [('A', 'B')]

Evaluate an Assignment
----------------------

``conin.common.log_potential`` evaluates a fully specified assignment for a
native CONIN Bayesian network or Markov network.

.. doctest:: quickstart

   >>> from conin.common import log_potential
   >>> value = log_potential(pgm, {"A": 0, "B": 1})
   >>> round(value, 6)
   np.float64(-0.328504)

Run Inference
-------------

For MAP inference on Bayesian networks, use one of the high-level inference
wrappers. Backend-specific requirements are listed in :doc:`backends`.

For example, with ``pgmpy`` installed:

.. code-block:: python

   from conin.inference import VariableEliminationInference

   result = VariableEliminationInference(pgm).map_query()
   print(result.solution.states)

Next Steps
----------

- See :doc:`probabilistic_graphical_models` for model-building guides.
- See :doc:`inference` for inference wrappers.
- See :doc:`examples` for selected example factories.
- See :doc:`model_conversion_io` for model conversion and file I/O.
- See :doc:`hmm_learning` for HMM parameter learning.
