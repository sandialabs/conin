Model Conversion and I/O
========================

``conin`` includes utilities for converting between model families, saving and
loading UAI files, and interoperating with optional probabilistic graphical model
libraries.

Core Conversions
----------------

Bayesian networks can be converted to Markov networks by transforming each CPD
into a factor.

.. code-block:: python

   from conin.bayesian_network import create_mn_from_bn
   from conin.bayesian_network.examples import simple1_BN_conin

   bn = simple1_BN_conin().pgm
   mn = create_mn_from_bn(bn)

   assert mn.nodes == ["A", "B"]
   assert mn.edges == [("A", "B")]

Hidden Markov models can be represented as dynamic Bayesian networks. Dynamic
Bayesian networks can then be unrolled into static Bayesian networks over a
specified inclusive time horizon. The HMM conversion creates dynamic variables
``H`` for hidden states and ``E`` for emitted observations.

.. code-block:: python

   from conin.dynamic_bayesian_network import create_bn_from_dbn
   from conin.hidden_markov_model import create_dbn_from_hmm
   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   dbn = create_dbn_from_hmm(hmm)
   bn = create_bn_from_dbn(dbn=dbn, start=0, stop=2)

   assert sorted(dbn.dynamic_states) == ["E", "H"]
   assert len(bn.nodes) == 6

UAI Files
---------

The ``conin.common`` namespace exposes a small unified I/O interface. For CONIN
models, ``save_model`` currently writes UAI files and ``load_model`` reads
``.uai`` and ``.uai.gz`` files.

Hidden Markov models use their own JSON methods, ``write_to_file`` and
``read_from_file``; see :doc:`hidden_markov_models` for an example.

.. code-block:: python

   from conin.bayesian_network.examples import simple1_BN_conin
   from conin.common import load_model, save_model

   pgm = simple1_BN_conin().pgm
   save_model(pgm, "simple.uai")
   loaded = load_model("simple.uai")

   assert loaded.nodes == ["var0", "var1"]

UAI files encode variables by integer position. When a model is loaded back into
CONIN, nodes are named ``var0``, ``var1``, and so on.

Model Utilities
---------------

``log_potential`` evaluates a fully specified assignment for a CONIN Markov or
Bayesian network. The current implementation requires values for every model
node; it does not marginalize over latent variables.

.. code-block:: python

   from conin.bayesian_network.examples import simple1_BN_conin
   from conin.common import is_polytree, log_potential

   pgm = simple1_BN_conin().pgm

   assert is_polytree(pgm)
   value = log_potential(pgm, {"A": 0, "B": 1})

``is_polytree`` accepts CONIN ``DiscreteBayesianNetwork`` instances and checks
whether the underlying undirected graph is singly connected.

Optional Backend Conversions
----------------------------

Additional conversion helpers are available when optional backend packages are
installed. See :doc:`backends` for a summary of available extras and backend
requirements.

``pgmpy`` support includes:

- ``conin.common.pgmpy.convert_pgmpy_to_conin``
- ``conin.common.conin.convert_conin_to_pgmpy_bn``
- ``conin.common.conin.convert_conin_to_pgmpy_mn``

Additional optional integrations include:

- ``conin.common.pgmpy.convert_pgmpy_to_pgmax``
- ``conin.common.pgmpy.convert_pgmpy_to_pomegranate``
- ``conin.common.load_model(..., model_type="pgmpy")``
- ``conin.common.load_model(..., model_type="pgmax")``
- ``conin.common.load_model(..., model_type="pomegranate")``
- ``conin.common.load_model(..., model_type="pyagrum")``

These conversions raise ``ImportError`` if the corresponding optional package is
not installed.

Reference
---------

.. autofunction:: conin.bayesian_network.create_mn_from_bn

.. autofunction:: conin.dynamic_bayesian_network.create_bn_from_dbn

.. autofunction:: conin.hidden_markov_model.create_dbn_from_hmm

.. autofunction:: conin.common.load_model

.. autofunction:: conin.common.save_model

.. autofunction:: conin.common.log_potential

.. autofunction:: conin.common.is_polytree

.. autofunction:: conin.common.conin.load_uai.load_conin_model_from_uai

.. autofunction:: conin.common.conin.save_model.save_model_uai

.. autofunction:: conin.common.conin.convert_conin_to_pgmpy_bn

.. autofunction:: conin.common.conin.convert_conin_to_pgmpy_mn

.. autofunction:: conin.common.pgmpy.convert_pgmpy_to_conin
