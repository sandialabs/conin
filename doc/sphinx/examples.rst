Selected Examples
=================

The ``conin`` package includes small example factories for each model family.
Each factory returns a ``munch.Munch`` object with a ``pgm`` field containing the
model and, when available, a ``solutions`` field with expected MAP/MPE results
used by the test suite.

This page highlights representative examples that are useful starting points for
new models. It is not an exhaustive API listing; inspect each model package's
``examples.py`` module for every available factory.

Backend Requirements
--------------------

Examples whose names end in ``_conin`` construct native CONIN models. Examples
whose names end in ``_pgmpy`` require ``pgmpy``. See :doc:`backends` for optional
package and solver requirements.

Markov Network Examples
-----------------------

Selected Markov network examples live in ``conin.markov_network.examples``.

.. list-table::
   :header-rows: 1

   * - Example
     - Description
   * - ``example6_conin``
     - Two-node Markov network with unary and pairwise factors.
   * - ``ABC_conin``
     - Three-variable Markov network with pairwise interactions.
   * - ``ABC2_conin``
     - Variant of ``ABC_conin`` with ordered expected solutions for all-optimal-solution testing.
   * - ``ABC_constrained_pyomo_conin``
     - Adds an all-different Pyomo constraint to ``ABC_conin``.
   * - ``ABC2_constrained_pyomo_conin``
     - Adds the same all-different Pyomo constraint to ``ABC2_conin``.
   * - ``ABC_constrained_toulbar2_conin``
     - Adds an all-different Toulbar2 constraint to ``ABC_conin``.
   * - ``ABC_constrained_oracle_conin``
     - Adds an all-different oracle constraint to ``ABC_conin``.
   * - ``ABC2_constrained_oracle_conin``
     - Adds an all-different oracle constraint to ``ABC2_conin``.
   * - ``example6_pgmpy``
     - ``pgmpy`` implementation of the two-node Markov network.
   * - ``ABC_pgmpy``
     - ``pgmpy`` implementation of the three-variable ``ABC`` network.
   * - ``ABC_constrained_pyomo_pgmpy``
     - Converts a ``pgmpy`` ``ABC`` model to CONIN and adds a Pyomo constraint.
   * - ``ABC_constrained_toulbar2_pgmpy``
     - Converts a ``pgmpy`` ``ABC`` model to CONIN and adds a Toulbar2 constraint.

Example usage:

.. code-block:: python

   from conin.markov_network.examples import ABC_conin

   example = ABC_conin()
   pgm = example.pgm
   expected = example.solutions[0].states

Bayesian Network Examples
-------------------------

Selected Bayesian network examples live in ``conin.bayesian_network.examples``.

.. list-table::
   :header-rows: 1

   * - Example
     - Description
   * - ``simple1_BN_conin``
     - Two-node Bayesian network with one conditional dependency.
   * - ``cancer1_BN_conin``
     - Five-node cancer network with explicit CPDs.
   * - ``DBDA_5_1_conin``
     - Three-node diagnostic example based on exercise 5.1 from *Doing Bayesian Data Analysis*.
   * - ``holmes_conin``
     - Six-node alarm/call Bayesian network adapted from lecture notes.
   * - ``tb2_BN_conin``
     - Three-node Bayesian network with non-binary state labels.
   * - ``cancer1_BN_constrained_pyomo_conin``
     - Adds Pyomo constraints to the cancer network.
   * - ``cancer1_BN_constrained_oracle_conin``
     - Adds an oracle constraint to the cancer network.
   * - ``cancer1_BN_constrained_toulbar2_conin``
     - Adds Toulbar2 constraints to the cancer network.
   * - ``*_pgmpy`` examples
     - ``pgmpy`` versions of selected Bayesian network examples.

Example usage:

.. code-block:: python

   from conin.bayesian_network.examples import simple1_BN_conin

   example = simple1_BN_conin()
   pgm = example.pgm
   expected = example.solutions[0].states

Dynamic Bayesian Network Examples
---------------------------------

Selected dynamic Bayesian network examples live in
``conin.dynamic_bayesian_network.examples``.

.. list-table::
   :header-rows: 1

   * - Example
     - Description
   * - ``simple0_DDBN_conin``
     - One dynamic node with an initial CPD and transition CPD.
   * - ``simple1_DDBN_conin``
     - Two dynamic nodes with intra-slice and transition dependencies.
   * - ``weather_conin``
     - Larger weather, temperature, observation, and humidity model with named states.
   * - ``simple1_DDBN_constrained_pyomo_conin``
     - Adds Pyomo constraints to ``simple1_DDBN_conin``.
   * - ``simple1_DDBN_constrained_oracle_conin``
     - Adds oracle constraints to ``simple1_DDBN_conin``.
   * - ``simple1_DDBN_constrained_toulbar2_conin``
     - Adds Toulbar2 constraints to ``simple1_DDBN_conin``.
   * - ``weather_constrained_pyomo_conin``
     - Adds a Pyomo constraint to the weather model.
   * - ``weather_constrained_oracle_conin``
     - Adds an oracle constraint to the weather model.
   * - ``weather_constrained_toulbar2_conin``
     - Adds a Toulbar2-style constraint to the weather model.
   * - ``*_pgmpy`` examples
     - ``pgmpy`` versions of selected dynamic Bayesian network examples.

Example usage:

.. code-block:: python

   from conin.dynamic_bayesian_network.examples import simple0_DDBN_conin

   example = simple0_DDBN_conin()
   dbn = example.pgm
   expected = example.solutions[0].states

Hidden Markov Model Examples
----------------------------

Selected hidden Markov model examples live in
``conin.hidden_markov_model.examples``.

.. list-table::
   :header-rows: 1

   * - Example
     - Description
   * - ``create_hmm0``
     - Deterministic two-state HMM.
   * - ``create_hmm1``
     - Compact two-state HMM used by the introductory HMM and inference docs.
   * - ``create_hmm1_aos``
     - Two-state HMM configured for all-optimal-solution testing.
   * - ``create_hmm2``
     - Three-state HMM with an absorbing third state.
   * - ``create_hmm2_aos``
     - Three-state HMM configured for all-optimal-solution testing.
   * - ``create_chmm1_oracle``
     - Constrained HMM using oracle sequence constraints.
   * - ``create_chmm1_oracle``
     - Constrained HMM using oracle constraints for variable-elimination workflows.
   * - ``create_chmm1_pyomo``
     - Constrained HMM using Pyomo constraints.
   * - ``create_chmm1_toulbar2``
     - Constrained HMM using Toulbar2 constraints.
   * - ``create_chmm1_pyomo_aos``
     - Pyomo-constrained HMM variant for all-optimal-solution testing.
   * - ``create_chmm2_pyomo_aos``
     - Three-state Pyomo-constrained HMM variant for all-optimal-solution testing.

Example usage:

.. code-block:: python

   from conin.hidden_markov_model.examples import create_hmm1

   hmm = create_hmm1()
   observed = ["o0", "o0", "o1"]
   hidden = hmm.generate_hidden(len(observed))
