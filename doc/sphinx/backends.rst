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
- ``ovld``
- ``smoek`` (algebraic constraints)

With only the base dependencies, users can create native CONIN models, work with
HMM utilities, declare every kind of constraint, and run HMM Viterbi/A*
inference.

.. list-table:: Base functionality
   :header-rows: 1

   * - Feature
     - Available with base dependencies
   * - Markov network, Bayesian network, dynamic Bayesian network, and HMM model construction
     - Yes
   * - Oracle, factor-oracle, recursive, Pyomo, Toulbar2, algebraic and MVR constraint declaration
     - Yes
   * - HMM generation with ``random_hmm`` and sampling with ``conin.sampling``
     - Yes (the MVR samplers need torch)
   * - HMM supervised learning and Monte Carlo EM
     - Yes
   * - HMM ``map_query`` with ``method="viterbi"`` or ``method="a_star"``
     - Yes
   * - ``log_probability`` and ``log_potential`` (``conin.scoring``)
     - Yes

Optional extras are declared in ``pyproject.toml``:

.. code-block:: shell

   python -m pip install -e .[pgmpy]     # variable elimination, pgmpy conversion, BIF files
   python -m pip install -e .[toulbar2]  # pytoulbar2
   python -m pip install -e .[torch]     # the MVR algorithms
   python -m pip install -e .[regex]     # greenery, for mvr_regex
   python -m pip install -e .[docs]      # Sphinx
   python -m pip install -e .[test]      # pytest

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

.. testcode::
   :skipif: not mip_solver_available

   from conin.config import default_mip_solver
   from conin.inference import map_query
   from conin.examples.markov_network import ABC_conin

   pgm = ABC_conin().pgm
   results = map_query(pgm, method="integer_program", solver=default_mip_solver)

If no supported solver is installed, Pyomo model construction may still work,
but optimization-based inference will not be able to solve the model.

``solver="or_topas"`` enumerates several optimal solutions with the
`or_topas <https://github.com/sandialabs/or_topas>`_ package, which is not
on PyPI; the conda environment files install it from GitHub. Its
``solver_options`` may include ``topas_method`` (``"balas"``, the default, or
``"gurobi_solution_pool"``).

Toulbar2
--------

Toulbar2-backed inference requires ``pytoulbar2`` (the ``toulbar2`` extra).

Use ``map_query(..., method="toulbar2")`` for Toulbar2-backed inference.

Toulbar2 CFN model files (``.cfn``, ``.cfn.gz``) can be loaded into CONIN
Markov networks with ``conin.io.load_model``. Linear constraints in these
files are loaded as Toulbar2 constraints. Loading does not require
``pytoulbar2``, but inference with the loaded constraints does; see
:doc:`model_conversion_io`.

When ``pytoulbar2`` is unavailable, ``map_query(..., method="toulbar2")``
returns an empty result object whose termination condition is
``"pytoulbar2 not available"``.

Smoek Algebraic Constraints
---------------------------

The ``algebraic_constraint_fn`` decorator uses ``smoek`` (a base dependency)
to let users write linear constraints with normal algebraic syntax. For
example, an HMM constraint can return expressions such as
``sum(model.V("H", t, "h0") for t in data.hmm.T) <= 12`` instead of mutating a
Pyomo model or Toulbar2 model directly.

Algebraic constraints can be used with:

- ``map_query(..., method="integer_program")``, which translates them to Pyomo.
- ``map_query(..., method="toulbar2")``, which translates supported linear
  constraints to Toulbar2.
- ``map_query(..., method="variable_elimination")`` and
  ``log_probability_query``, which compile them into factor-oracle constraints.

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
- ``conin.scoring.log_probability_query``
- Loading and saving BIF files through ``conin.io.load_model`` and
  ``conin.io.save_model``
- ``conin.io.load_model(..., model_type="pgmpy")``
- ``conin.io.save_model(..., model_type="pgmpy")``
- ``conin.common.convert_pgmpy_to_conin``
- ``conin.common.convert_conin_to_pgmpy_bn``
- ``conin.common.convert_conin_to_pgmpy_mn``

Variable elimination runs in pgmpy on the (unrolled) model, so its memory use
grows quickly with the size of the largest intermediate factor. For example,
MAP inference on the unconstrained ``weather_conin`` dynamic Bayesian network
unrolled to ``stop=4`` has run out of memory in our tests; use
``integer_program`` or ``toulbar2`` for such models.

torch (MVR algorithms)
----------------------

The MVR algorithms in ``conin.inference.mvr``, ``conin.learning.mvr`` and
``conin.sampling.mvr``, and
``map_query(..., method="mvr_viterbi")``, require ``torch`` (the ``torch``
extra). torch is imported the first time one of these algorithms is used:
``import conin`` and importing these packages do not import it. Declaring MVR
constraints (``conin.constraints.mvr``) does not need torch, but
``mvr_regex`` needs ``greenery`` (the ``regex`` extra).

Development Environments
------------------------

The repository also includes conda environment files:

- ``environment.yml`` for a runtime-style environment.
- ``dev_environment.yml`` for development and testing.

Both include ``pytoulbar2``, ``highspy``, ``smoek`` and ``or_topas`` through
pip. The development environment additionally includes ``pytest``,
``pytest-cov``, ``nbmake`` and ``nbstripout``.

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
   * - Run variable elimination inference or ``log_probability_query``
     - ``pgmpy``
   * - Use algebraic constraints
     - Base package plus the selected inference backend
   * - Convert to/from pgmpy models
     - ``pgmpy``
   * - Run integer-programming inference
     - Pyomo plus ``gurobi``, ``highs``, or ``glpk``
   * - Enumerate several optimal solutions
     - ``or_topas`` plus a MIP solver
   * - Run Toulbar2/CFN inference
     - ``pytoulbar2``
   * - Load Toulbar2 CFN files
     - Base package
   * - Run the MVR algorithms
     - ``torch`` (``greenery`` for ``mvr_regex``)
