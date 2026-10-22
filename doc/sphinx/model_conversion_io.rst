Model Conversion and I/O
========================

``conin`` includes utilities for converting between model families, saving and
loading UAI files, and interoperating with optional probabilistic graphical model
libraries.

Core Conversions
----------------

Bayesian networks can be converted to Markov networks by transforming each CPD
into a factor.

.. testcode::

   from conin.graphical_models import create_mn_from_bn
   from conin.examples.bayesian_network import simple1_BN_conin

   bn = simple1_BN_conin().pgm
   mn = create_mn_from_bn(bn)

   assert mn.nodes == ["A", "B"]
   assert mn.edges == [("A", "B")]

Hidden Markov models can be represented as dynamic Bayesian networks. Dynamic
Bayesian networks can then be unrolled into static Bayesian networks over a
specified inclusive time horizon. The HMM conversion creates dynamic variables
``H`` for hidden states and ``E`` for emitted observations.

.. testcode::

   from conin.graphical_models import create_bn_from_dbn
   from conin.graphical_models import create_dbn_from_hmm
   from conin.examples.hidden_markov_model import create_hmm1

   hmm = create_hmm1()
   dbn = create_dbn_from_hmm(hmm)
   bn = create_bn_from_dbn(dbn=dbn, start=0, stop=2)

   assert sorted(dbn.dynamic_states) == ["E", "H"]
   assert len(bn.nodes) == 6

``create_bn_from_dbn`` unrolls from ``start=0`` (the DBN's initial CPDs
describe time 0; another ``start`` raises ``InvalidInputError``) to ``stop``
inclusive; ``stop=0`` gives one time step. The ``start`` argument of
``map_query`` with ``integer_program``, ``toulbar2`` and
``variable_elimination`` on DBNs and HMMs follows the same rule.

UAI Files
---------

The ``conin.io`` namespace exposes a small unified I/O interface. For CONIN
models, ``save_model`` currently writes UAI files and ``load_model`` reads
``.uai``, ``.uai.gz``, ``.cfn`` and ``.cfn.gz`` files.

Hidden Markov models use their own JSON methods, ``write_to_file`` and
``read_from_file``; see :doc:`hidden_markov_models` for an example.

.. testcode::

   from conin.examples.bayesian_network import simple1_BN_conin
   from conin.io import load_model, save_model

   pgm = simple1_BN_conin().pgm
   save_model(pgm, "simple.uai")
   loaded = load_model("simple.uai")

   assert loaded.nodes == ["var0", "var1"]

UAI files encode variables by integer position. When a model is loaded back into
CONIN, nodes are named ``var0``, ``var1``, and so on.

Toulbar2 CFN Files
------------------

``load_model`` also reads Toulbar2 cost function network files (``.cfn`` and
``.cfn.gz``) into a CONIN ``DiscreteMarkovNetwork``. This does not require
``pytoulbar2``. Variable names and value names are taken from the file.

Each cost function becomes a factor with values ``exp(-cost / cost_scale)``,
so the most probable assignment of the Markov network is an optimal solution
of the CFN. Entries that violate the CFN upper bound (``"mustbe": "<UB"``)
get zero factor values. CFN files that Toulbar2 writes from probabilistic
models store costs as negative log-probabilities multiplied by ``10**7``, so
load them with ``cost_scale=1e7``:

.. testsetup:: cfn

   import os
   import shutil
   import conin.io

   _cfn_dir = os.path.join(os.path.dirname(conin.io.__file__), "tests")
   shutil.copy(os.path.join(_cfn_dir, "cancer_mn.cfn"), "model.cfn")
   shutil.copy(os.path.join(_cfn_dir, "abc_constrained.cfn"), "constrained_model.cfn")

.. testcode:: cfn

   from conin.io import load_model

   pgm = load_model("model.cfn", cost_scale=1e7)

Hard linear constraints are also loaded. Toulbar2 stores these as ``knapsack``
and ``knapsackv`` global cost functions; they are written by pytoulbar2's
``AddLinearConstraint``, ``AddSumConstraint`` and
``AddGeneralizedLinearConstraint`` methods. When a file contains such
constraints, ``load_model`` returns a ``ConstrainedDiscreteMarkovNetwork``
whose constraints are Toulbar2 constraints, and the unconstrained model is
available as its ``model`` attribute:

.. testcode:: cfn
   :skipif: not toulbar2_available

   from conin.io import load_model
   from conin.inference import map_query

   cpgm = load_model("constrained_model.cfn")
   constrained = map_query(cpgm, method="toulbar2")
   unconstrained = map_query(cpgm.model, method="toulbar2")

Loaded constraints can only be used with ``method="toulbar2"``. Each one also
stores a description of its linear constraint (``terms``, ``operator`` and
``rhs``) so that it can be converted to other constraint types later.

Other global cost functions, such as all-different or regular-language
constraints, raise ``NotImplementedError``. Hard entries in cost tables are
loaded as zero-valued factor entries, not as CONIN constraints.

Model Utilities
---------------

``log_probability`` and ``log_potential`` evaluate a total assignment: every
variable has a value, so nothing is summed out.

- ``log_probability(model, assignment, *, observed=None)`` returns ``log P(x)``
  for a Bayesian network, a dynamic Bayesian network (``{(name, t): state}``
  over the times ``0..stop``) or an HMM (the hidden states, plus the
  ``observed`` states). It raises ``InvalidInputError`` for Markov networks and
  constrained models, whose probabilities need an inference query.
- ``log_potential`` takes the same arguments. It also accepts Markov networks,
  where it returns the sum of the log factors (``log P(x) + log Z``), and
  constrained models, where it returns ``-inf`` if a constraint is violated.
  Constraints of the ``"oracle"`` and ``"recursive"`` (HMMs), ``"factor_oracle"``,
  ``"algebraic"`` and ``"mvr"`` (HMMs) families are checked with the models'
  ``is_feasible`` method;
  ``"pyomo"`` and ``"toulbar2"`` constraints raise ``NotImplementedError``.

pgmpy models are converted to CONIN models first.

``log_probability_query(model, assignment, *, evidence=None,
method="variable_elimination", start=0, stop=None)`` computes a marginal or
conditional log-probability, ``log P(X = x | E = e)``, where the variables in
neither ``assignment`` nor ``evidence`` are summed out. For a constrained
model it is the probability under the constraints, ``log P(X = x | E = e, C)``
(``C``: every constraint holds), and Markov networks are normalised. It returns
a float (``-inf`` for probability zero) and raises ``InvalidInputError`` if the
evidence (with the constraints) has probability zero.

- ``assignment`` and ``evidence`` are dicts. The variables of a dynamic
  Bayesian network are ``(name, t)``; those of an HMM are ``("H", t)`` (hidden)
  and ``("E", t)`` (observed), so ``{("E", t): o, ...}`` with no evidence is
  the likelihood of the observations.
- Dynamic models are unrolled over ``start..stop``; ``stop`` defaults to the
  largest time in the query (at least ``start + 1``). With constraints, it sets
  the time steps the constraints see.
- ``method="variable_elimination"`` (the only method, which needs pgmpy) runs on
  the ``"factor"`` representation: unconstrained models and the
  ``"factor_oracle"`` and ``"algebraic"`` constraint families. Other families
  raise ``InvalidInputError``.

.. testcode::
   :skipif: not pgmpy_available

   from conin.examples.bayesian_network import (
       cancer1_BN_conin,
       cancer1_BN_constrained_oracle_conin,
   )
   from conin.scoring import log_probability_query

   pgm = cancer1_BN_conin().pgm
   cpgm = cancer1_BN_constrained_oracle_conin().pgm
   log_probability_query(pgm, {"Cancer": 1}, evidence={"Xray": 1})
   log_probability_query(cpgm, {"Cancer": 1})  # log P(Cancer=1 | constraints)

.. doctest:: model_utilities

   >>> from conin.examples.bayesian_network import (
   ...     cancer1_BN_constrained_oracle_conin,
   ...     simple1_BN_conin,
   ... )
   >>> from conin.common import is_polytree
   >>> from conin.scoring import log_potential, log_probability
   >>> pgm = simple1_BN_conin().pgm
   >>> is_polytree(pgm)
   True
   >>> round(log_probability(pgm, {"A": 0, "B": 1}), 6)
   -0.328504
   >>> cpgm = cancer1_BN_constrained_oracle_conin().pgm
   >>> x = {"Cancer": 1, "Dyspnoea": 0, "Pollution": 0, "Smoker": 1, "Xray": 1}
   >>> cpgm.is_feasible(x), round(log_potential(cpgm, x), 6)
   (True, -1.890152)
   >>> cpgm.is_feasible({**x, "Xray": 0}), log_potential(cpgm, {**x, "Xray": 0})
   (False, -inf)

``is_polytree`` accepts CONIN ``DiscreteBayesianNetwork`` instances and checks
whether the underlying undirected graph is singly connected.

Optional Backend Conversions
----------------------------

Additional conversion helpers are available when optional backend packages are
installed. See :doc:`backends` for a summary of available extras and backend
requirements.

``pgmpy`` support includes:

- ``conin.common.pgmpy.convert_pgmpy_to_conin``
- ``conin.common.pgmpy.convert_conin_to_pgmpy_bn``
- ``conin.common.pgmpy.convert_conin_to_pgmpy_mn``
- ``conin.io.load_model(..., model_type="pgmpy")``
- ``conin.io.save_model(..., model_type="pgmpy")``

These conversions raise ``ImportError`` if the corresponding optional package is
not installed.

Reference
---------

The functions on this page are documented in :doc:`api_scoring_io`.
