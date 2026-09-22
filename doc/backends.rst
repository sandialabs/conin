Backend and Optional Dependencies
=================================

Many ``conin`` features work with the base package dependencies, while some
inference and interoperability features require optional packages or external
solvers. This page summarizes what is needed for each backend.

Base Package
------------

The base package dependencies are declared in ``pyproject.toml``:

- ``munch``
- ``numpy``
- ``pandas``
- ``pyomo``

With only the base dependencies, users can create native CONIN models, work with
HMM utilities, use oracle, factor, Pyomo, and Toulbar2 constraint declarations,
and run HMM Viterbi/A* inference. Algebraic constraints additionally require
``smoek``.

.. list-table:: Base functionality
   :header-rows: 1

   * - Feature
     - Available with base dependencies
   * - Markov network, Bayesian network, dynamic Bayesian network, and HMM model construction
     - Yes
   * - Oracle and factor constraint declaration
     - Yes
   * - Algebraic constraint declaration with ``algebraic_constraint_fn``
     - Requires ``smoek``
   * - HMM generation with ``random_hmm``
     - Yes
   * - HMM supervised learning
     - Yes
   * - HMM ``map_query`` with ``method="viterbi"`` or ``method="a_star"``
     - Yes

Pyomo Solvers
-------------

``pyomo`` is a base dependency, but Pyomo-based inference also needs an
available optimization solver. ``conin`` checks for these solvers when selecting
its default MIP solver:

- ``gurobi``
- ``highs``
- ``glpk``

Use ``map_query(..., method="integer_program")`` for Pyomo-backed inference.

You can also pass a solver explicitly to ``map_query``:

.. code-block:: python

   from conin.inference import map_query
   from conin.markov_network.examples import ABC_conin

   pgm = ABC_conin().pgm
   results = map_query(pgm, method="integer_program", solver="highs")

If no supported solver is installed, Pyomo model construction may still work,
but optimization-based inference will not be able to solve the model.

Toulbar2
--------

Toulbar2-backed inference requires ``pytoulbar2`` and a working Toulbar2
installation. The conda environment files include ``pytoulbar2`` as a pip
dependency, but it is not currently exposed as a Python package extra in
``pyproject.toml``.

Use ``map_query(..., method="toulbar2")`` for Toulbar2-backed inference.

When ``pytoulbar2`` is unavailable, some static-model Toulbar2 helpers return
an empty result object with a termination condition indicating that
``pytoulbar2`` is not available. Dynamic Bayesian network and HMM Toulbar2
inference paths should be treated as requiring ``pytoulbar2`` at runtime.

Smoek Algebraic Constraints
---------------------------

The ``algebraic_constraint_fn`` decorator uses ``smoek`` to let users write
linear constraints with normal algebraic syntax. For example, an HMM constraint
can return expressions such as ``sum(model.V("H", t, "h0") for t in data.hmm.T)
<= 12`` instead of mutating a Pyomo model or Toulbar2 model directly.

``smoek`` is required when algebraic constraints are applied by the inference
backends. The conda environment files install it from the Sandia GitHub
repository:

.. code-block:: shell

   python -m pip install -e git+https://github.com/sandialabs/smoek.git#egg=smoek

Algebraic constraints can be used with:

- ``map_query(..., method="integer_program")``, which translates them to Pyomo.
- ``map_query(..., method="toulbar2")``, which translates supported linear
  constraints to Toulbar2.

The selected inference method still needs its own backend dependencies, such as
a Pyomo-supported MIP solver for ``integer_program`` or ``pytoulbar2`` for
``toulbar2``.

pgmpy
-----

``pgmpy`` is optional and can be installed with:

.. code-block:: shell

   python -m pip install -e .[pgmpy]

Features requiring ``pgmpy`` include:

- ``map_query(..., method="variable_elimination")``
- Loading BIF files through ``conin.common.load_model(..., model_type="conin")``
- ``conin.common.load_model(..., model_type="pgmpy")``
- ``conin.common.save_model(..., model_type="pgmpy")``
- ``conin.common.pgmpy.convert_pgmpy_to_conin``
- ``conin.common.conin.convert_conin_to_pgmpy_bn``
- ``conin.common.conin.convert_conin_to_pgmpy_mn``

Other Optional Integrations
---------------------------

Additional extras are declared for model-loading and conversion integrations:

The ``pyagrum``, ``pgmax``, and ``pomegranate`` loaders use ``pgmpy`` as an
intermediate representation, so those extras include ``pgmpy`` in addition to
the target package.

.. list-table:: Optional integration extras
   :header-rows: 1

   * - Extra
     - Package
     - Related functionality
   * - ``pyagrum``
     - ``pgmpy``, ``pyagrum``
     - ``conin.common.load_model(..., model_type="pyagrum")``
   * - ``pgmax``
     - ``pgmpy``, ``pgmax``
     - ``conin.common.load_model(..., model_type="pgmax")`` and ``convert_pgmpy_to_pgmax``
   * - ``pomegranate``
     - ``pgmpy``, ``pomegranate``
     - ``conin.common.load_model(..., model_type="pomegranate")`` and ``convert_pgmpy_to_pomegranate``
   * - ``pgm-all``
     - ``pgmpy``, ``pyagrum``, ``pgmax``, ``pomegranate``
     - Installs the declared optional PGM integration packages

For example:

.. code-block:: shell

   python -m pip install -e .[pgm-all]

Development Environments
------------------------

The repository also includes conda environment files:

- ``environment.yml`` for a runtime-style environment.
- ``dev_environment.yml`` for development and testing.

Both include ``pytoulbar2`` and ``highspy`` through pip. The development
environment additionally includes ``pytest`` and ``pytest-cov``.

Quick Selection Guide
---------------------

.. list-table:: Which backend do I need?
   :header-rows: 1

   * - Goal
     - Dependency
   * - Build CONIN models and examples
     - Base package
   * - Run HMM Viterbi/A* inference
     - Base package
   * - Run supervised HMM learning
     - Base package
   * - Run variable elimination inference
     - ``pgmpy``
   * - Use algebraic constraints
     - ``smoek`` plus the selected inference backend
   * - Convert to/from pgmpy models
     - ``pgmpy``
   * - Run integer-programming inference
     - Pyomo plus ``gurobi``, ``highs``, or ``glpk``
   * - Run Toulbar2/CFN inference
     - ``pytoulbar2``/Toulbar2
   * - Use pgmax, pomegranate, or pyagrum integrations
     - Corresponding optional extra, which also installs ``pgmpy``
